"""Model loading for the eval pod.

`load_model(model_ref)` returns an object satisfying the `Model` protocol
(`generate(prompt) -> str`). Two backends:

- `stub:<flavor>` — a CPU-only deterministic stand-in (no torch/GPU/download),
  used for the local toy. Flavors: `perfect`, `sycophantic`, `weak`.
- a filesystem path / HF id — Phase 2: load a real HF model. Left as a clear
  TODO so the toy stays importable without torch installed.
"""
from __future__ import annotations

import os
import re

_FACTS = {
    "capital of france": "paris",
    "gas do plants release": "oxygen",
    "chemical symbol for gold": "au (gold)",
    "planet is closest to the sun": "mercury",
    "largest ocean": "pacific",
}


def _arith(a: int, op: str, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def _as_messages(prompt) -> list[dict]:
    """Normalize a prompt to a chat-messages list. A `str` -> ONE user turn (so every
    string-prompt benchmark is byte-identical to before). A `list[dict]` (role/content
    turns) passes through, so a benchmark can send a real SYSTEM / multi-turn prompt
    (MASK's pressure prompt; prompt-injection). Anything else -> str()."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


def _flatten_messages(messages: list[dict]) -> str:
    """Collapse a messages list into one string for chat templates that reject a system
    role / consecutive same-role turns (e.g. Gemma-2). Content is preserved verbatim
    (system first, then user); only the role packaging is dropped."""
    return "\n\n".join((m.get("content") or "").strip()
                       for m in messages if (m.get("content") or "").strip())


def _ensure_str(x) -> str:
    """apply_chat_template(tokenize=False) returns a `str` on transformers <=4.x but a
    1-element `list[str]` on transformers 5.x for a single conversation. Coerce to the
    rendered string so the tokenizer always receives clean `str` / `list[str]`."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return _ensure_str(x[0]) if x else ""
    return str(x)


class StubModel:
    """Deterministic, dependency-free model for the toy harness.

    Parses the prompt enough to answer arithmetic + a few facts. The `flavor`
    controls behavior under "are you sure?" pressure so different flavors get
    different trajectory scores:
      - perfect:      recomputes and holds its correct answer under pressure
      - sycophantic:  correct first turn, but caves to the asserted distractor
      - weak:         gets ~half the base questions wrong
    """

    def __init__(self, flavor: str = "perfect"):
        self.flavor = flavor

    def candidate_logits(self, prompt: str, candidates: list[str],
                         use_chat_template: bool = True) -> list[float]:
        """Toy heuristic: no real logits. Returns uniform 0.0 for every
        candidate (so wei -> P(No)=0.5, MMLU -> argmax=first). Real scoring uses
        HFModel; this just keeps logprob benchmarks from crashing under a stub."""
        return [0.0 for _ in candidates]

    # Batched variants — trivial for the stub (just map).
    def generate_batch(self, prompts: list[str], **kw) -> list[str]:
        return [self.generate(p, **kw) for p in prompts]

    def candidate_logits_batch(self, prompts: list[str], candidates: list[str],
                               use_chat_template: bool = True) -> list[list[float]]:
        return [self.candidate_logits(p, candidates, use_chat_template) for p in prompts]

    def completion_logprob_batch(self, prompts: list[str], completions: list[str],
                                 use_chat_template: bool = False) -> list[float]:
        """Stub: uniform 0.0 (length-normalized logprob). Keeps FACTOR runnable under
        the toy — argmax over equal scores picks index 0 (the factual completion)."""
        return [0.0 for _ in prompts]

    def generate(self, prompt, **_: object) -> str:
        if not isinstance(prompt, str):           # accept a messages list (MASK etc.)
            prompt = _flatten_messages(_as_messages(prompt))
        is_pressure = "are you sure" in prompt.lower()

        # Under pressure, the sycophantic flavor caves to the asserted distractor.
        if is_pressure and self.flavor == "sycophantic":
            m = re.search(r"answer is (-?\d+)", prompt)
            if m:
                return m.group(1)

        # Arithmetic: solve the LAST equation in the prompt (the question line).
        eqs = re.findall(r"(\d+)\s*([+\-*])\s*(\d+)", prompt)
        if eqs:
            a, op, b = eqs[0]  # the question is the first equation in the prompt
            ans = _arith(int(a), op, int(b))
            if self.flavor == "weak" and (int(a) + int(b)) % 2 == 0:
                ans += 1  # deterministically wrong on ~half
            return str(ans)

        # Facts.
        low = prompt.lower()
        for key, val in _FACTS.items():
            if key in low:
                if self.flavor == "weak":
                    return "i'm not sure"
                return val
        return "i don't know"


class HFModel:
    """Real model backend: a local HF directory (merged weights or base+adapter)
    wrapped to satisfy the `Model` protocol. torch/transformers are imported
    lazily so this module stays importable in CPU-only / toy contexts."""

    def __init__(self, model_path: str, max_new_tokens: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            # Some instruct LLMs ship as a multimodal model (e.g. Mistral3 — "an LLM
            # with vision"); the text tokenizer lives behind an AutoProcessor.
            from transformers import AutoProcessor
            self.tokenizer = AutoProcessor.from_pretrained(model_path).tokenizer
        # Left-padding so batched generation + last-token logits work uniformly
        # (the real last token sits at position -1 for every row).
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        _dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=_dtype).to(self.device)
        except (ValueError, KeyError):
            # A multimodal-architecture LLM (e.g. Mistral3ForConditionalGeneration — a strong
            # text LLM that also takes images) is NOT an AutoModelForCausalLM. Load it via the
            # image-text-to-text class and generate TEXT-ONLY (no pixel_values) — for our rule/
            # judge benches (pure-text prompts) the generation path is identical. Only triggers
            # when the causal-LM load fails, so text-only models are unaffected.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=_dtype).to(self.device)
        self.model.eval()
        # Generation budget. Priority: explicit arg > EVAL_MAX_NEW_TOKENS env >
        # AUTO. AUTO (the default) = the model's FULL remaining context per prompt,
        # i.e. the max this model can produce — so a fixed cap can never silently
        # truncate a benchmark's outputs again. (This is the bug that broke GSM8K:
        # a 256-token cap cut CoT before the final answer, scoring correct work as
        # wrong and putting the base model below its own floor.) EOS still bounds
        # real cost, so "max per model" is safe in practice; truncation is also
        # surfaced (see _note_truncation) so it can never be silent.
        env_mnt = os.getenv("EVAL_MAX_NEW_TOKENS")
        self.max_new_tokens = (max_new_tokens if max_new_tokens is not None
                               else (int(env_mnt) if env_mnt else None))  # None = AUTO
        # Model context window for the AUTO budget; guard tokenizer sentinels (1e30).
        ctx = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if ctx <= 0 or ctx > 1_000_000:
            tml = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
            ctx = tml if 0 < tml <= 1_000_000 else 4096
        self._ctx = ctx
        # AUTO budget = remaining context, but capped at EVAL_AUTO_CEILING so a
        # DEGENERATE model (one that never emits EOS — common for broken fine-tunes
        # the AAR produces) can't make a single row generate the model's full 32k
        # context and stall the whole eval. The ceiling is set comfortably above
        # any legitimate answer (GSM8K CoT / IFEval ≈ a few hundred tokens), so it
        # never truncates real outputs; raise it via env if a suite ever needs more.
        self._auto_ceiling = int(os.getenv("EVAL_AUTO_CEILING", "2048"))
        self._trunc_count = 0
        # --- robust stop tokens (fixes "won't stop" / wrong-EOS) -----------------
        # A chat model ends its TURN with a turn-ender token that is NOT always the
        # model's nominal eos. Classic traps: Llama-3 turns end with <|eot_id|> (not
        # <|end_of_text|>); Qwen3.x turns end with <|im_end|> but the tokenizer update
        # made <|endoftext|> the eos, so generate() stops on the WRONG token and the
        # model rambles past its turn (verified: Qwen3.5-2B gen-config eos=248044
        # <|endoftext|> != turn-ender 248046 <|im_end|>). We therefore stop on the
        # UNION of every known stop id — the loaded generation_config, the model
        # config, AND the tokenizer's eos. This covers the turn-ender for every model
        # (verified per-model vs HF) and can only stop EARLIER, never later (all are
        # turn-end-only special tokens). gemma's nominal eos (<eos>=1) is NOT its
        # turn-ender (106) — the union keeps 106 from gen-config, so it stays correct.
        def _as_ids(x):
            if x is None:
                return []
            seq = x if isinstance(x, (list, tuple)) else [x]
            return [int(i) for i in seq if i is not None]
        _stop = set(_as_ids(getattr(self.model.generation_config, "eos_token_id", None)))
        _stop |= set(_as_ids(getattr(self.model.config, "eos_token_id", None)))
        if self.tokenizer.eos_token_id is not None:
            _stop.add(int(self.tokenizer.eos_token_id))
        self._stop_ids = sorted(_stop)
        self._eos = self.tokenizer.eos_token_id  # legacy; stopping now uses _stop_ids
        print(f"[eval] stop token ids = {self._stop_ids} "
              f"(tokenizer.eos={self.tokenizer.eos_token_id})", flush=True)
        self.batch_size = int(os.getenv("EVAL_BATCH_SIZE", "16"))
        # Anti-degeneration guard (opt-in, default OFF so it never silently changes
        # other evals). Greedy decoding can fall into a repetition loop that never
        # emits EOS and so runs to the budget → "truncation". Blocking repeated
        # n-grams forces a stuck generation to move on and terminate, eliminating
        # those loop-truncations. Set EVAL_NO_REPEAT_NGRAM (e.g. 4) to enable; must
        # be applied UNIFORMLY to baseline + trained-model evals to stay comparable.
        self._no_repeat_ngram = int(os.getenv("EVAL_NO_REPEAT_NGRAM", "0"))
        # Decoding strategy. Greedy (temperature=0) is deterministic but degenerates into
        # repetition loops on some models (they never emit EOS → runaway to the budget).
        # SAMPLING (temperature>0) follows the model's real distribution and terminates
        # normally. Default temperature=1.0 (full sampling). EVAL_SEED makes it REPRODUCIBLE
        # (seeded before every generate), so baseline + trained-model evals stay comparable.
        self._temperature = float(os.getenv("EVAL_TEMPERATURE", "1.0"))
        self._top_p = float(os.getenv("EVAL_TOP_P", "1.0"))
        self._seed = int(os.getenv("EVAL_SEED", "1234"))
        self._torch = torch

    def apply_decoding(self, *, batch_size=None, auto_ceiling=None,
                       no_repeat_ngram=None, max_new_tokens="__keep__",
                       temperature=None, top_p=None, seed=None) -> None:
        """Reconfigure decoding for the NEXT benchmark(s). Called per-benchmark by
        run_eval so each is scored at the EXACT golden config its baseline was
        measured at (benchmark_docs) — a single eval run bundles benchmarks from
        different baseline groups (the shared capability basket vs the axis's own),
        which a single global setting cannot serve. Only provided knobs change;
        max_new_tokens='__keep__' leaves it untouched, None restores AUTO."""
        if batch_size is not None:
            self.batch_size = int(batch_size)
        if auto_ceiling is not None:
            self._auto_ceiling = int(auto_ceiling)
        if no_repeat_ngram is not None:
            self._no_repeat_ngram = int(no_repeat_ngram)
        if max_new_tokens != "__keep__":
            self.max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else None
        if temperature is not None:
            self._temperature = float(temperature)
        if top_p is not None:
            self._top_p = float(top_p)
        if seed is not None:
            self._seed = int(seed)

    def _gen_extra(self) -> dict:
        """Extra HF generate() kwargs shared by generate/generate_batch — the decoding
        STRATEGY. temperature>0 => sampling (the model's real distribution; terminates
        normally); temperature==0 => greedy. Seed each call for reproducibility."""
        self._torch.manual_seed(self._seed)
        extra: dict = {}
        if self._no_repeat_ngram > 0:
            extra["no_repeat_ngram_size"] = self._no_repeat_ngram
        if self._temperature and self._temperature > 0:
            extra.update(do_sample=True, temperature=self._temperature, top_p=self._top_p)
        else:
            extra["do_sample"] = False
        return extra

    def _budget(self, input_len: int, override=None) -> int:
        """Max new tokens for this call. AUTO (max_new_tokens is None) = the model's
        remaining context capped at the AUTO ceiling — never a small fixed cap, so
        real outputs are never truncated, but a runaway model is still bounded."""
        if override is not None:
            return int(override)
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return max(64, min(self._ctx - int(input_len) - 16, self._auto_ceiling))

    def _note_truncation(self, new_tokens, mnt: int) -> None:
        """Surface — never silence — a completion that filled the whole budget
        without ever emitting a stop token (genuinely truncated). Works for both
        single and left-padded batched generation (finished rows contain eos/pad)."""
        try:
            row = new_tokens.tolist() if hasattr(new_tokens, "tolist") else list(new_tokens)
        except Exception:
            return
        if len(row) < mnt:
            return  # stopped on its own — not truncated
        stop_ids = set(self._stop_ids) | ({self.tokenizer.pad_token_id} - {None})
        if not any(t in stop_ids for t in row):
            self._trunc_count += 1
            if self._trunc_count <= 5 or self._trunc_count % 50 == 0:
                print(f"[eval] WARNING: a completion hit the {mnt}-token budget with no "
                      f"stop token (truncated x{self._trunc_count}); raw output is cut off "
                      f"so answer-extraction may fail. Raise the generation budget.", flush=True)

    def generate(self, prompt, **kwargs) -> str:
        # Render the prompt (a string = one user turn, identical to before; or a messages
        # list = real system/multi-turn) through _render.
        text = self._render(prompt, True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        mnt = self._budget(enc["input_ids"].shape[1], kwargs.get("max_new_tokens"))
        with self._torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=mnt,
                eos_token_id=self._stop_ids or None,   # stop on the turn-ender (union; fixes Qwen <|im_end|>)
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **self._gen_extra(),   # supplies do_sample/temperature/top_p (or greedy)
            )
        new = out[0, enc["input_ids"].shape[1]:]
        self._note_truncation(new, mnt)
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    def candidate_logits(self, prompt: str, candidates: list[str],
                         use_chat_template: bool = True) -> list[float]:
        """Continuation log-likelihood for each candidate after the prompt (see
        candidate_logits_batch — correct for multi-token candidates like Phi's ' A')."""
        return self.candidate_logits_batch([prompt], candidates, use_chat_template)[0]

    # --- batched variants (the GPU throughput win) -------------------------
    def _render(self, prompt, use_chat_template: bool = True) -> str:
        """Render a prompt to input text. `prompt` is a string (one user turn, byte-identical
        to before) OR a messages list (real system/multi-turn). If the chat template rejects
        the roles (no system / consecutive same-role, e.g. Gemma-2), flatten to one user turn.
        apply_chat_template's output is coerced to `str` (transformers 5.x returns a 1-elem list)."""
        if not use_chat_template:
            return prompt if isinstance(prompt, str) else _flatten_messages(_as_messages(prompt))
        messages = _as_messages(prompt)
        try:
            return _ensure_str(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            try:   # roles rejected (e.g. Gemma system) -> flatten to one user turn
                return _ensure_str(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": _flatten_messages(messages)}],
                    tokenize=False, add_generation_prompt=True))
            except Exception:
                return _flatten_messages(messages)

    def generate_batch(self, prompts: list[str], **kw) -> list[str]:
        """Generate for many prompts at once (chunked by batch_size). ~batch_size×
        faster than looping generate()."""
        override = kw.get("max_new_tokens")
        outs: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = [self._render(p, True) for p in prompts[i:i + self.batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            in_len = enc["input_ids"].shape[1]
            mnt = self._budget(in_len, override)
            with self._torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=mnt,
                    eos_token_id=self._stop_ids or None,   # stop on the turn-ender (union; fixes Qwen <|im_end|>)
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **self._gen_extra())   # supplies do_sample/temperature/top_p (or greedy)
            new = gen[:, in_len:]
            for r in range(new.shape[0]):
                self._note_truncation(new[r], mnt)
            outs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
        return [o.strip() for o in outs]

    def candidate_logits_batch(self, prompts: list[str], candidates: list[str],
                               use_chat_template: bool = True) -> list[list[float]]:
        """Score each candidate by its CONTINUATION log-likelihood (sum of its token
        logprobs given the prompt), and return one score per candidate per prompt.

        This replaces first-token-logit scoring, which is WRONG for tokenizers that split a
        candidate into multiple tokens: e.g. Phi tokenizes ' A'/' B'/' C'/' D' as
        [space,'A']/[space,'B']/... so the *first* token (the shared space) is identical for
        all four → all logits equal → argmax collapses to A (~random). Summing the full
        continuation makes the shared leading space cancel in the argmax while the letter
        token discriminates. For single-token candidates this is argmax-identical to the old
        behaviour (logprob is monotonic in the logit at the same position), so the models
        whose ' A' is one token are unaffected."""
        torch = self._torch
        cand_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in candidates]
        ncand = len(candidates)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        # One (prompt + candidate) sequence per (prompt, candidate); score them in batches.
        flat: list[list[int]] = []
        meta: list[tuple[int, int, int, list[int]]] = []  # (prompt_idx, cand_idx, prompt_len, cand_ids)
        for pi, p in enumerate(prompts):
            p_ids = self.tokenizer(self._render(p, use_chat_template)).input_ids
            for ci, ids in enumerate(cand_ids):
                flat.append(list(p_ids) + list(ids))
                meta.append((pi, ci, len(p_ids), ids))
        results: list[list[float]] = [[float("-inf")] * ncand for _ in prompts]
        for b in range(0, len(flat), self.batch_size):
            seqs = flat[b:b + self.batch_size]
            mb = meta[b:b + self.batch_size]
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to(self.device); attn = attn.to(self.device)
            with torch.no_grad():
                lp = torch.log_softmax(self.model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
            for r, (pi, ci, plen, ids) in enumerate(mb):
                if not ids:
                    continue
                # token ids[j] is predicted at position (plen-1+j); right-pad keeps these valid.
                results[pi][ci] = float(sum(lp[r, plen - 1 + j, tid].item() for j, tid in enumerate(ids)))
        return results

    def completion_logprob_batch(self, prompts: list[str], completions: list[str],
                                 use_chat_template: bool = False) -> list[float]:
        """LENGTH-NORMALIZED continuation log-prob (mean token log-prob) of each
        `completions[i]` given `prompts[i]` — one float per index. Matches AI21's
        FACTOR `eval_factuality.py`, which sums token NLL over the completion span
        (prefix masked) then divides by completion length; FACTOR picks argmax over
        the 4 completions = the factual one. `use_chat_template` defaults False:
        FACTOR is a raw-text continuation task (no chat wrapper), like the paper."""
        torch = self._torch
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        flat: list[list[int]] = []
        meta: list[tuple[int, int, list[int]]] = []   # (idx, prompt_len, completion_ids)
        for i, (p, c) in enumerate(zip(prompts, completions)):
            ctx = self._render(p, use_chat_template)
            # JOINT tokenization (matches AI21 eval_factuality.py: tokenize prefix+completion as
            # one sequence and mask the prefix) — so the prefix/completion boundary tokenizes
            # NATURALLY (separate tokenize(prefix)+tokenize(completion) splits the boundary token
            # for BPE tokenizers and biases the logprob). Completion span = the joint tokens after
            # the prefix; back off plen on the rare boundary-merge so positions stay aligned.
            ctx_ids = self.tokenizer(ctx).input_ids
            full_ids = self.tokenizer(ctx + c).input_ids
            plen = len(ctx_ids)
            while plen > 0 and full_ids[:plen] != ctx_ids[:plen]:
                plen -= 1
            c_ids = full_ids[plen:]
            flat.append(list(full_ids))
            meta.append((i, plen, c_ids))
        out: list[float] = [float("-inf")] * len(prompts)
        for b in range(0, len(flat), self.batch_size):
            seqs = flat[b:b + self.batch_size]
            mb = meta[b:b + self.batch_size]
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to(self.device); attn = attn.to(self.device)
            with torch.no_grad():
                lp = torch.log_softmax(self.model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
            for r, (i, plen, c_ids) in enumerate(mb):
                if not c_ids:
                    continue
                # c_ids[j] is predicted at position (plen-1+j); right-pad keeps these valid.
                tot = sum(lp[r, plen - 1 + j, tid].item() for j, tid in enumerate(c_ids))
                out[i] = float(tot / len(c_ids))     # mean token log-prob (length-normalized)
        return out


def load_model(model_ref: str):
    """Return a Model for `model_ref`.
    - `stub:<flavor>` -> StubModel (CPU toy, no deps)
    - any other string -> a local filesystem path to an HF model dir -> HFModel
    """
    if model_ref.startswith("stub:"):
        return StubModel(flavor=model_ref.split(":", 1)[1] or "perfect")
    return HFModel(model_ref)
