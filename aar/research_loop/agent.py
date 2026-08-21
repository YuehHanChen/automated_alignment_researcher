"""
Autonomous W2S research agent using Claude Agent SDK.

Merges agent_loop, base_agent, and stop_conditions into a single file.
"""

import asyncio
import json
import os
import re
import shutil
import time
import traceback
import uuid
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from aar.config import (
    WORKSPACE_DIR,
    S3_BUCKET,
    S3_IDEAS_PREFIX,
    LOGS_DIR,
    FULL_AUTO_MAX_RUNTIME_SECONDS,
    LOCAL_FINDINGS_DIR,
    FINDINGS_POLL_INTERVAL,
    TARGET_IDEA_FILE,
    DATASET_NAME,
    DATA_DIR,
    WEAK_MODEL,
    STRONG_MODEL,
    SERVER_URL,
    AAR_MODE,
)
from .tools.server_api_tools import create_server_api_tools_server
from .tools.prior_work_tools import create_prior_work_tools_server
from .tools.findings_sync import FindingsSync


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _format_baselines() -> str:
    """Render the measured base-model baselines into a prompt-ready block, read
    from the authoritative baselines.json (so the prompt never carries stale
    hardcoded numbers — the bug that broke the gsm8k floor). Empty if unavailable."""
    import os
    path = os.getenv("BASELINES_PATH")
    if not path:
        try:
            from aar.config import HARNESS_RUNS_DIR
            path = os.path.join(HARNESS_RUNS_DIR, "baselines.json")
        except Exception:
            return ""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return ""
    # HELD-OUT benchmarks (the axis's generalization set, e.g. sycon_fp) are INVISIBLE
    # to the AAR — never render them in the prompt, or the AAR could optimize toward
    # them. The launcher sets HELD_OUT_BENCH (comma-sep) for the axis's held-out set.
    held = {b.strip() for b in os.getenv("HELD_OUT_BENCH", "").split(",") if b.strip()}
    lines = []
    for name, d in data.items():
        if name in held:
            continue
        b = d.get("baseline")
        fl = d.get("floor")
        s = f"    - `{name}`: base model scores **{b}**"
        if fl is not None:
            s += f"  — capability floor **{fl}** (your method's CI must not fall clearly below this)"
        lines.append(s)
    return "\n".join(lines)


def resolve_prompt(template_path: str | Path, output_path: str | Path) -> str:
    """Render Jinja2 prompt template with config variables."""
    from jinja2 import Template

    template_path = Path(template_path)
    output_path = Path(output_path)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    target_idea_content = ""
    if Path(TARGET_IDEA_FILE).exists():
        with open(TARGET_IDEA_FILE, 'r') as f:
            try:
                idea_data = json.load(f)
                target_idea_content = idea_data.get("Description", json.dumps(idea_data, indent=2))
            except json.JSONDecodeError:
                target_idea_content = f.read()
    else:
        print(f"WARNING: Target idea file not found: {TARGET_IDEA_FILE}")

    local_mode = os.getenv("LOCAL_MODE", "false").lower() in ("1", "true", "yes")

    # Generalized harness vars (suite/target) alongside the legacy W2S ones, so
    # either prompt template renders.
    from aar.config import SUITE_NAME, TARGET_MODEL
    content = template.render(
        workspace_dir=WORKSPACE_DIR,
        dataset_name=DATASET_NAME,
        data_dir=DATA_DIR,
        weak_model=WEAK_MODEL,
        strong_model=STRONG_MODEL,
        suite_name=SUITE_NAME,
        # Per-axis (swappable via launcher env) — keeps the prompt axis-agnostic.
        safety_property=os.getenv("SAFETY_PROPERTY", SUITE_NAME),
        objective=os.getenv("SAFETY_OBJECTIVE", f"improve {SUITE_NAME}"),
        # OPT-IN full-observability briefing (2026-06-21): when BENCHMARK_BRIEF_FILE is set, its contents
        # (a detailed explanation of the scored benchmarks + metrics + scoring) render into the prompt.
        # EMPTY by default -> isolated runs are unchanged (D2 preserved). Used for the privacy
        # observability experiment; pairs with the per-iteration score decomposition (BenchmarkScore.extra).
        benchmark_brief=(open(os.environ["BENCHMARK_BRIEF_FILE"]).read()
                         if os.getenv("BENCHMARK_BRIEF_FILE") and os.path.exists(os.environ["BENCHMARK_BRIEF_FILE"])
                         else ""),
        seed_method=os.getenv("SEED_METHOD", ""),   # axis's reference seed idea dir (optional)
        target_model=TARGET_MODEL,
        s3_bucket=S3_BUCKET,
        s3_ideas_prefix=S3_IDEAS_PREFIX,
        logs_dir=LOGS_DIR,
        # Per-chain memory file + chain id (parallel chains must NOT share an
        # AGENT_LOG or a method-dir namespace). agent_log_path is seeded by
        # slurm_aar_chain.sh with an explicit "iteration 1, no history" stub.
        agent_log_path=os.getenv("AAR_AGENT_LOG_PATH", f"{LOGS_DIR}/AGENT_LOG.md"),
        chain_id=(re.sub(r"[^A-Za-z0-9]+", "_", os.getenv("IDEA_NAME", "chain")).strip("_") or "chain"),
        # Decoupled (GPU-less agent) mode: the chain holds no GPU, so the agent must
        # train via a separate low-QoS job. Default false => original inline-capable prompt.
        no_local_gpu=os.getenv("AAR_NO_LOCAL_GPU", "false").lower() in ("1", "true", "yes"),
        server_url=SERVER_URL,
        aar_mode=str(AAR_MODE).lower(),
        local_mode=str(local_mode).lower(),
        target_idea_content=target_idea_content,
        baselines_table=_format_baselines(),
    )

    with open(output_path, 'w') as f:
        f.write(content)

    return content


# ---------------------------------------------------------------------------
# Stop condition (just timeout)
# ---------------------------------------------------------------------------

class StopReason(Enum):
    TIMEOUT = "timeout"
    USER_INTERRUPT = "user_interrupt"


class _StopChecker:
    """Simple timeout-only stop checker."""

    def __init__(self, max_runtime: float):
        self.max_runtime = max_runtime
        self.start_time = time.time()
        self.consecutive_errors = 0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def check(self) -> Optional[StopReason]:
        if self.elapsed_time >= self.max_runtime:
            return StopReason.TIMEOUT
        return None

    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    success: bool
    output: Dict[str, Any]
    duration: float
    iteration_count: int
    error: Optional[str] = None


def _is_overloaded(err: object) -> bool:
    """True if an exception/message is an Anthropic API 529 'Overloaded' error.

    These are transient capacity errors — the right response is to wait and
    retry, not to abandon the iteration. We match on the text so it works
    whether the SDK raises a typed error or surfaces it as a generic message.
    """
    s = str(err).lower()
    return "overloaded" in s or "529" in s


# How long to wait, and how many times to retry, when the API is overloaded.
# Defaults: wait 10 minutes, up to 6 times (~1h of riding out an outage) before
# giving up on the session. Override via env for tests / different SLAs.
OVERLOADED_WAIT_SECONDS = int(os.getenv("OVERLOADED_WAIT_SECONDS", "600"))
OVERLOADED_MAX_RETRIES = int(os.getenv("OVERLOADED_MAX_RETRIES", "6"))


# ---------------------------------------------------------------------------
# Base agent (wraps Claude Agent SDK)
# ---------------------------------------------------------------------------

class BaseAgent:
    """Wraps ClaudeSDKClient for research tasks."""

    def __init__(
        self,
        name: str,
        allowed_tools: List[str],
        workspace: Path,
        mcp_servers: Dict[str, Any],
        model: str = os.getenv("AAR_AGENT_MODEL", "claude-opus-4-8"),
        permission_mode: str = "bypassPermissions",
        cli_path: Optional[str] = None,
        message_callback: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.allowed_tools = allowed_tools
        self.workspace = workspace
        self.mcp_servers = mcp_servers
        self.model = model
        self.permission_mode = permission_mode
        self.cli_path = cli_path
        self.message_callback = message_callback
        self.system_prompt = system_prompt

    async def execute(self, task: str) -> AgentResult:
        """Execute agent task. Returns AgentResult.

        If the API returns a 529 'Overloaded' error, wait OVERLOADED_WAIT_SECONDS
        (default 10 min) and retry the same session, up to OVERLOADED_MAX_RETRIES
        times — a capacity blip shouldn't burn an iteration or kill the chain.
        """
        start_time = time.time()
        overloaded_retries = 0

        while True:
            iteration_count = 0
            messages = []
            try:
                return await self._execute_once(task, start_time)
            except Exception as e:
                if _is_overloaded(e) and overloaded_retries < OVERLOADED_MAX_RETRIES:
                    overloaded_retries += 1
                    print(
                        f"[{self.name}] API overloaded (529). Waiting "
                        f"{OVERLOADED_WAIT_SECONDS}s before retry "
                        f"{overloaded_retries}/{OVERLOADED_MAX_RETRIES}..."
                    )
                    await asyncio.sleep(OVERLOADED_WAIT_SECONDS)
                    continue
                return AgentResult(
                    success=False,
                    output={},
                    duration=time.time() - start_time,
                    iteration_count=0,
                    error=str(e),
                )

    async def _execute_once(self, task: str, start_time: float) -> AgentResult:
        """One attempt at running the session (no retry). Raises on API error so
        execute() can decide whether to retry (529) or give up."""
        iteration_count = 0
        messages = []

        if True:
            options_dict = {
                "allowed_tools": self.allowed_tools,
                "system_prompt": self.system_prompt,
                "permission_mode": self.permission_mode,
                "cwd": str(self.workspace),
                "model": self.model,
                "mcp_servers": self.mcp_servers,
                "setting_sources": ["project"],
                "betas": ["context-1m-2025-08-07"],
                # Extended thinking. Claude 4.x+ models (fable-5, opus-4.8) use ADAPTIVE thinking and
                # default the thinking `display` to "omitted" -> the model thinks but the summary is
                # withheld (empty ThinkingBlock.thinking, signature only -> empty [THINKING] in logs).
                # Set display="summarized" to surface the reasoning summary the session-log callback
                # persists. Manual `max_thinking_tokens` (=> thinking type "enabled"/budget_tokens) is
                # REJECTED on these models — depth is controlled by `effort` instead (default "high";
                # bump to xhigh/max via AAR_EFFORT).
                "thinking": {"type": "adaptive", "display": "summarized"},
                "effort": os.getenv("AAR_EFFORT", "max"),
            }
            if self.cli_path:
                options_dict["cli_path"] = self.cli_path

            options = ClaudeAgentOptions(**options_dict)

            async with ClaudeSDKClient(options=options) as client:
                await client.query(task)

                async for message in client.receive_response():
                    messages.append(message)

                    # Log
                    log_msg = self._format_message(message)
                    if log_msg:
                        print(log_msg)

                    if self.message_callback:
                        self.message_callback(message)

                    if isinstance(message, ResultMessage):
                        break

                    if isinstance(message, AssistantMessage):
                        for content in message.content:
                            if isinstance(content, ToolUseBlock):
                                iteration_count += 1

                return AgentResult(
                    success=True,
                    output=self._extract_output(messages),
                    duration=time.time() - start_time,
                    iteration_count=iteration_count,
                )

    def _format_message(self, message) -> Optional[str]:
        ts = datetime.now().strftime("%H:%M:%S")
        if isinstance(message, AssistantMessage):
            parts = []
            for content in message.content:
                if isinstance(content, TextBlock):
                    parts.append(f"[{ts}] [{self.name}] {content.text[:200]}")
                elif isinstance(content, ToolUseBlock):
                    tool_input = content.input or {}
                    detail = ""
                    if content.name == "Bash":
                        detail = f" {tool_input.get('command', '')[:100]}"
                    elif content.name in ("Read", "Write", "Edit"):
                        detail = f" {tool_input.get('file_path', '')}"
                    parts.append(f"[{ts}] [{self.name}] -> {content.name}{detail}")
            return "\n".join(parts) if parts else None
        elif isinstance(message, ResultMessage):
            return f"[{ts}] [{self.name}] Done"
        return None

    def _extract_output(self, messages) -> Dict[str, Any]:
        output = {"text_outputs": [], "tool_uses": []}
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for content in msg.content:
                    if isinstance(content, TextBlock):
                        output["text_outputs"].append(content.text)
                    elif isinstance(content, ToolUseBlock):
                        output["tool_uses"].append({"tool": content.name, "input": content.input})
            if isinstance(msg, ResultMessage):
                output["result_message"] = msg.result
        return output


# ---------------------------------------------------------------------------
# Main autonomous loop
# ---------------------------------------------------------------------------

class AutonomousAgentLoop:
    """
    Main agent loop for autonomous W2S research.

    Each iteration is a fresh Claude session. Agent reads/writes findings.json
    directly. Only stop condition: timeout.
    """

    def __init__(
        self,
        idea_uid: str,
        idea_name: str,
        workspace: Optional[Path] = None,
        max_runtime_seconds: Optional[int] = None,
        max_iterations: Optional[int] = None,
        logs_dir: Optional[Path] = None,
        s3_bucket: Optional[str] = None,
        model: str = os.getenv("AAR_AGENT_MODEL", "claude-opus-4-8"),
        local_mode: bool = False,
    ):
        if not idea_uid:
            raise ValueError("idea_uid is required")

        self.idea_uid = idea_uid
        self.idea_name = idea_name
        self.local_mode = local_mode
        self.run_id = os.getenv("RUN_ID") or str(uuid.uuid4())

        self.workspace = Path(workspace) if workspace else Path(WORKSPACE_DIR)
        self.logs_dir = Path(logs_dir) if logs_dir else Path(LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.logs_dir.parent / "findings.json"

        self.max_runtime_seconds = max_runtime_seconds or FULL_AUTO_MAX_RUNTIME_SECONDS
        # Cap on number of iterations (= sessions). env MAX_ITERATIONS or arg; None = unbounded.
        self.max_iterations = max_iterations if max_iterations is not None else (
            int(os.getenv("MAX_ITERATIONS")) if os.getenv("MAX_ITERATIONS") else None)
        self.s3_bucket = s3_bucket or S3_BUCKET
        self.s3_prefix = f"{S3_IDEAS_PREFIX}{idea_uid}/{self.run_id}/"
        self.model = model

        self.stop_checker = _StopChecker(max_runtime=self.max_runtime_seconds)

        # Create MCP servers (server API tools always needed — in local mode, server runs on localhost)
        self.mcp_servers = {}
        try:
            self.mcp_servers["server-api-tools"] = create_server_api_tools_server()
        except Exception as e:
            print(f"[Init] Warning: server API tools unavailable: {e}")
        if not local_mode:
            try:
                self.mcp_servers["prior-work-tools"] = create_prior_work_tools_server()
            except Exception as e:
                print(f"[Init] Warning: prior work tools unavailable: {e}")

        # Findings sync: disabled in local mode (no other workers)
        self.findings_sync = None
        if not local_mode:
            self.findings_sync = FindingsSync(
                findings_dir=Path(LOCAL_FINDINGS_DIR),
                poll_interval=FINDINGS_POLL_INTERVAL,
            )

        self.session_count = 0
        self._prompt: Optional[str] = None

    def _get_prompt(self) -> str:
        if self._prompt is None:
            # PROMPT_TEMPLATE selects the system prompt (e.g. prompt_safety.jinja2,
            # the generic axis-agnostic submit-model prompt; prompt.jinja2 for legacy W2S).
            tmpl_name = os.getenv("PROMPT_TEMPLATE", "prompt.jinja2")
            template_path = self.workspace / "aar" / "research_loop" / tmpl_name
            if not template_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {template_path}")
            resolved_path = self.logs_dir / "prompt_resolved.md"
            self._prompt = resolve_prompt(template_path, resolved_path)
        return self._prompt

    def _create_agent(self, session_id: str, message_callback=None) -> BaseAgent:
        allowed_tools = [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebSearch", "WebFetch",
            "mcp__server-api-tools__evaluate_model",
            "mcp__server-api-tools__evaluate_predictions",
            "mcp__server-api-tools__share_finding",
            "mcp__server-api-tools__get_leaderboard",
            "mcp__server-api-tools__get_literature",
            "mcp__server-api-tools__share_literature",
            "mcp__server-api-tools__submit_idea_proposal",
        ]
        if not self.local_mode:
            allowed_tools.append("mcp__prior-work-tools__download_snapshot")

        return BaseAgent(
            name=f"autonomous-{session_id}",
            allowed_tools=allowed_tools,
            workspace=self.workspace,
            mcp_servers=self.mcp_servers,
            model=self.model,
            cli_path=shutil.which("claude"),
            message_callback=message_callback,
        )

    async def run(self) -> Dict[str, Any]:
        """Run the autonomous agent loop."""
        mode_str = "local" if self.local_mode else "server"
        print(f"\n{'='*60}")
        print(f"Autonomous Agent Loop ({mode_str} mode)")
        print(f"  Run ID: {self.run_id}")
        print(f"  Idea: {self.idea_name} ({self.idea_uid})")
        print(f"  Max Runtime: {self.max_runtime_seconds/3600:.1f}h")
        print(f"  Max Iterations: {self.max_iterations if self.max_iterations else 'unbounded'}")
        print(f"{'='*60}\n")

        # Initial findings sync (disabled in local mode)
        if self.findings_sync:
            try:
                count = self.findings_sync.sync_once()
                print(f"[Loop] Fetched {count} finding(s) from server")
            except Exception as e:
                print(f"[Loop] Warning: initial findings fetch failed: {e}")

            try:
                self.findings_sync.start()
            except Exception as e:
                print(f"[Loop] Warning: could not start findings sync: {e}")

        stop_reason = None

        while True:
            if self.max_iterations and self.session_count >= self.max_iterations:
                print(f"\n[Loop] Stopping: reached max iterations ({self.max_iterations})")
                break
            stop_reason = self.stop_checker.check()
            if stop_reason:
                print(f"\n[Loop] Stopping: {stop_reason.value}")
                break

            try:
                await self._run_session()
                self.session_count += 1
                self.stop_checker.record_success()
                if not self.local_mode:
                    await self._sync_to_s3()

            except KeyboardInterrupt:
                print("\n[Loop] Interrupted")
                stop_reason = StopReason.USER_INTERRUPT
                break

            except Exception as e:
                print(f"\n[Loop] Session error: {e}")
                traceback.print_exc()
                self.stop_checker.record_error()
                # If a 529 still reached the loop (e.g. after in-call retries
                # were exhausted), back off the full 10 min rather than 30s so we
                # don't hammer an overloaded API and burn sessions.
                backoff = OVERLOADED_WAIT_SECONDS if _is_overloaded(e) else 30
                await asyncio.sleep(backoff)

        if self.findings_sync:
            try:
                self.findings_sync.stop()
            except Exception:
                pass

        if not self.local_mode:
            await self._sync_to_s3()

        result = {
            "run_id": self.run_id,
            "idea_uid": self.idea_uid,
            "idea_name": self.idea_name,
            "sessions": self.session_count,
            "duration_seconds": self.stop_checker.elapsed_time,
            "stop_reason": stop_reason.value if stop_reason else "unknown",
        }

        print(f"\n{'='*60}")
        print(f"Done: {result['sessions']} sessions, "
              f"{result['duration_seconds']/3600:.1f}h, "
              f"reason={result['stop_reason']}")
        print(f"{'='*60}\n")

        return result

    async def _run_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Chain identity in the session id: parallel chains that start in the SAME
        # second otherwise collide on ONE session_<ts>.log (observed: 4 chains ->
        # one transcript, so the dashboard showed identical trajectories for all).
        _tag = re.sub(r"[^A-Za-z0-9]+", "_", self.idea_name or "chain").strip("_") or "chain"
        session_id = f"session_{self.session_count:03d}_{_tag}_{timestamp}"
        log_file = self.logs_dir / f"{session_id}.log"

        print(f"\n[Session {self.session_count}] {session_id}")

        prompt = self._get_prompt()

        with open(log_file, "w") as log_f:
            log_f.write(f"# Session {session_id}\n# Started: {datetime.now().isoformat()}\n\n")

            def message_callback(message):
                ts = datetime.now().strftime("%H:%M:%S")
                log_f.write(f"\n[{ts}] {type(message).__name__}\n")
                if hasattr(message, "content"):
                    for content in message.content:
                        if hasattr(content, "thinking"):   # ThinkingBlock — RAW extended reasoning
                            log_f.write(f"[THINKING]\n{content.thinking}\n")
                        elif hasattr(content, "text"):
                            log_f.write(f"{content.text}\n")
                        elif hasattr(content, "name"):
                            log_f.write(f"Tool: {content.name}\n")
                            if hasattr(content, "input"):
                                input_str = json.dumps(content.input, indent=2)
                                if len(input_str) > 1000:
                                    input_str = input_str[:1000] + "... [truncated]"
                                log_f.write(f"Input: {input_str}\n")
                log_f.flush()

            try:
                agent = self._create_agent(session_id, message_callback=message_callback)
                result = await agent.execute(task=prompt)

                log_f.write(f"\n# Result: success={result.success}, "
                           f"duration={result.duration:.1f}s, "
                           f"tools={result.iteration_count}\n")
                if result.error:
                    log_f.write(f"# Error: {result.error}\n")

                if not result.success:
                    raise RuntimeError(f"Agent failed: {result.error}")

            except Exception as e:
                log_f.write(f"\n# ERROR: {e}\n{traceback.format_exc()}")
                raise
            finally:
                log_f.write(f"\n# Ended: {datetime.now().isoformat()}\n")

        print(f"[Session {self.session_count}] Completed")

    async def _sync_to_s3(self):
        try:
            from aar.infrastructure.s3_utils import upload_file_to_s3
        except ImportError:
            return

        try:
            if self.findings_path.exists():
                upload_file_to_s3(
                    file_path=self.findings_path,
                    s3_key=f"{self.s3_prefix}findings.json",
                    bucket_name=self.s3_bucket,
                    content_type="application/json",
                )
            for log_file in self.logs_dir.glob("session_*.log"):
                upload_file_to_s3(
                    file_path=log_file,
                    s3_key=f"{self.s3_prefix}logs/{log_file.name}",
                    bucket_name=self.s3_bucket,
                    content_type="text/plain",
                )
        except Exception as e:
            print(f"[Sync] Failed: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main():
    import sys
    idea_uid = os.getenv("IDEA_UID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    idea_name = os.getenv("IDEA_NAME", "unknown")
    local_mode = os.getenv("LOCAL_MODE", "false").lower() in ("1", "true", "yes")

    if not idea_uid:
        print("Usage: python -m aar.research_loop.agent <idea_uid>")
        print("   Or: IDEA_UID=... python -m aar.research_loop.agent")
        sys.exit(1)

    loop = AutonomousAgentLoop(idea_uid=idea_uid, idea_name=idea_name, local_mode=local_mode)
    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
