#!/usr/bin/env python3
"""
Unified launcher for weak-to-strong research.

Usage:
    # List available ideas
    python run.py list
    
    # Run an idea locally (single seed)
    python run.py --idea vanilla_w2s --seed 42

    # Run with multiple seeds across available GPUs
    python run.py --idea vanilla_w2s --seeds 42,43,44,45,46

    # Run agent in local mode (server on localhost, no S3)
    python run.py agent --idea-uid <uid> --idea-name <name> --local

    # Start web dashboard (also handles RunPod deployment)
    python run.py server
"""
import argparse
import importlib
import os
import sys
import subprocess
from pathlib import Path


def cmd_agent(args, remaining):
    """Launch autonomous research agent."""
    import asyncio

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is required for agent mode")
        sys.exit(1)

    local_mode = getattr(args, 'local', False)

    if local_mode:
        # Local mode: server runs on localhost, no S3, no findings sync
        os.environ.setdefault("ORCHESTRATOR_API_URL", "http://localhost:8000")
        os.environ["LOCAL_MODE"] = "true"
        print("Local mode: using server at http://localhost:8000")
        print("Make sure the server is running: python run.py server")

    from aar.research_loop.agent import AutonomousAgentLoop

    loop = AutonomousAgentLoop(
        idea_uid=args.idea_uid,
        idea_name=args.idea_name or "unknown",
        max_runtime_seconds=args.max_runtime,
        max_iterations=getattr(args, "max_iterations", None),
        model=args.model,
        local_mode=local_mode,
    )
    asyncio.run(loop.run())


def cmd_server(args, remaining):
    """Start web dashboard."""
    server_dir = Path(__file__).parent / "aar" / "web_ui" / "backend"
    if not server_dir.exists():
        # Try new path
        server_dir = Path(__file__).parent / "aar" / "server"
    if not server_dir.exists():
        print("Error: server directory not found")
        sys.exit(1)

    port = args.port or 8000
    print(f"Starting server on port {port}...")
    subprocess.run(
        [sys.executable, "app.py"],
        cwd=str(server_dir),
        env={**os.environ, "PORT": str(port)},
    )


def cmd_list(args=None, remaining=None):
    """List available ideas."""
    _list_ideas()


def _list_ideas():
    """Print available ideas."""
    ideas_dir = Path(__file__).parent / "aar" / "ideas"
    if not ideas_dir.exists():
        print("No ideas directory found")
        return

    ideas = sorted(
        d.name for d in ideas_dir.iterdir()
        if d.is_dir()
        and (d / "run.py").exists()
        and not d.name.startswith("_")
        and not d.name.startswith("lmgen_")
    )
    print("Available ideas:")
    for idea in ideas:
        print(f"  {idea}")


def _print_results(results):
    """Print experiment results."""
    if not results:
        return
    print("\n" + "=" * 60)
    print("Results:")
    if results.get("aar_mode"):
        print(f"  [AAR Mode] Predictions: {len(results.get('predictions', []))} samples")
    else:
        for key in ["weak_acc", "transfer_acc", "strong_acc", "pgr"]:
            if results.get(key) is not None:
                print(f"  {key}: {results[key]:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Weak-to-Strong Research Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py agent --idea-uid abc123 --idea-name "my idea" --local
  python run.py server
  python run.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Agent subcommand
    agent_parser = subparsers.add_parser("agent", help="Run autonomous research agent")
    agent_parser.add_argument("--idea-uid", required=True, help="Idea UID")
    agent_parser.add_argument("--idea-name", default=None, help="Idea name")
    agent_parser.add_argument("--max-iterations", type=int, default=None,
                              help="Stop after this many iterations (= sessions)")
    agent_parser.add_argument("--max-runtime", type=int, default=5*24*3600,
                              help="Max runtime in seconds (default: 5 days)")
    agent_parser.add_argument("--model", default=os.getenv("AAR_AGENT_MODEL", "claude-opus-4-8"), help="Claude model")
    agent_parser.add_argument("--local", action="store_true",
                              help="Local mode: server on localhost, no S3/findings sync")

    # Server subcommand
    server_parser = subparsers.add_parser("server", help="Start web dashboard")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # List subcommand
    subparsers.add_parser("list", help="List available ideas")

    args, remaining = parser.parse_known_args()

    # Route to subcommand
    if args.command == "agent":
        cmd_agent(args, remaining)
    elif args.command == "server":
        cmd_server(args, remaining)
    elif args.command == "list":
        cmd_list(args, remaining)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python run.py agent --idea-uid <uid> --idea-name <name> --local")
        print("  python run.py list                           # See available method dirs")


if __name__ == "__main__":
    main()
