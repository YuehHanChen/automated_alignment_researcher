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

    def generate(self, prompt, **_: object) -> str:
        if isinstance(prompt, list):   # messages list -> flatten for the stub's text heuristics
            prompt = "\n".join(m.get("content", "") for m in prompt)
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Left-padding so batched generation + last-token logits work uniformly
        # (the real last token sits at position -1 for every row).
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
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
        self._eos = self.tokenizer.eos_token_id
        self.batch_size = int(os.getenv("EVAL_BATCH_SIZE", "16"))
        # Anti-degeneration guard (opt-in, default OFF so it never silently changes
        # other evals). Greedy decoding can fall into a repetition loop that never
        # emits EOS and so runs to the budget → "truncation". Blocking repeated
        # n-grams forces a stuck generation to move on and terminate, eliminating
        # those loop-truncations. Set EVAL_NO_REPEAT_NGRAM (e.g. 4) to enable; must
        # be applied UNIFORMLY to baseline + trained-model evals to stay comparable.
        self._no_repeat_ngram = int(os.getenv("EVAL_NO_REPEAT_NGRAM", "0"))
        self._torch = torch

    def _gen_extra(self) -> dict:
        """Extra HF generate() kwargs shared by generate/generate_batch."""
        return {"no_repeat_ngram_size": self._no_repeat_ngram} if self._no_repeat_ngram > 0 else {}

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
        stop_ids = {self._eos, self.tokenizer.pad_token_id} - {None}
        if not any(t in stop_ids for t in row):
            self._trunc_count += 1
            if self._trunc_count <= 5 or self._trunc_count % 50 == 0:
                print(f"[eval] WARNING: a completion hit the {mnt}-token budget with no "
                      f"stop token (truncated x{self._trunc_count}); raw output is cut off "
                      f"so answer-extraction may fail. Raise the generation budget.", flush=True)

    def generate(self, prompt, **kwargs) -> str:
        # `prompt` is a string (wrapped as one user turn — the default for almost every
        # benchmark) OR a list of {role, content} messages, so a benchmark can use the
        # source paper's real system / multi-turn structure (InjecAgent SYS_PROMPT,
        # Tensor Trust pre_prompt-as-system + attack/post user turns).
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt if isinstance(prompt, str) else "\n".join(m.get("content", "") for m in messages)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        mnt = self._budget(enc["input_ids"].shape[1], kwargs.get("max_new_tokens"))
        with self._torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=mnt,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **self._gen_extra(),
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
    def _render(self, prompt, use_chat_template: bool) -> str:
        # `prompt` may be a string (one user turn) or a list of {role, content} messages
        # (the source paper's real system / multi-turn structure).
        flat = prompt if isinstance(prompt, str) else "\n".join(m.get("content", "") for m in prompt)
        if not use_chat_template:
            return flat
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return flat

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
                    **enc, max_new_tokens=mnt, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **self._gen_extra())
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


def load_model(model_ref: str):
    """Return a Model for `model_ref`.
    - `stub:<flavor>` -> StubModel (CPU toy, no deps)
    - any other string -> a local filesystem path to an HF model dir -> HFModel
    """
    if model_ref.startswith("stub:"):
        return StubModel(flavor=model_ref.split(":", 1)[1] or "perfect")
    return HFModel(model_ref)
