"""Template for a new method (a.k.a. "idea") in the multi-benchmark AAR harness.

==============================================================================
THE CONTRACT (submit-model)
==============================================================================
A method implements:

    run_experiment(config) -> {"model_path": <dir-or-ref>, "method": <name>}

It TRAINS a model and returns where the trained model lives. It does NOT score
the model and does NOT emit predictions. The harness uploads `model_path` and a
held-out eval pod scores it against the secret benchmark suite, returning only
aggregate scores.

==============================================================================
HARD RULES (the isolation boundary — do not cross)
==============================================================================
- You NEVER see the benchmark test inputs or answers. They live only on the
  eval pod. There is no "predictions" path. If your method needs benchmark
  items to work, it is the wrong method.
- Train only on ALLOWED data: open-source corpora you download (UltraChat,
  HH-RLHF, OASST, ARC, SciQ, ...) or synthetic data you generate yourself.
- Do NOT import anything from `aar.benchmarks` or the eval pod, and do
  NOT read anything under the secret data prefix.
- One model submission per iteration; the eval pod is the only path to scores.

Everything routing-specific (paths, base model, hyperparams) comes from
`config`. Heavy deps (torch/transformers) are imported lazily inside
run_experiment so this module stays importable in CPU-only / toy contexts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MethodConfig:
    """Minimal config a method needs. Extend per-method as required.
    (In the full harness this is constructed from CLI/env like W2S's RunConfig.)"""
    base_model: str = "Qwen/Qwen1.5-0.5B-Chat"
    output_dir: str = "results/template/model"
    seed: int = 42
    dry_run: bool = False  # toy/smoke: skip training, return a stub model ref


def load_base_model_and_tokenizer(model_id: str, **kw):
    """Load a base model + tokenizer for training. Loads with
    ``attn_implementation="eager"`` — ALWAYS use this (or pass it yourself).

    WHY EAGER (do not remove): some target architectures — notably Phi-3
    (``Phi3ForCausalLM``) — do NOT support the default SDPA / FlashAttention-2
    backends in the installed transformers and CRASH during any generation-in-the-loop
    training (self-play, on-policy DPO, rejection sampling) with errors like
    ``Phi3 does not support ...scaled_dot_product_attention`` or
    ``'DynamicCache' object has no attribute 'seen_tokens'``. ``eager`` is the
    universally-supported attention path — correct for EVERY architecture, only modestly
    slower than SDPA — so your method trains robustly across all target models with no
    per-model special-casing. (Saving the tokenizer alongside the model is also required
    so the eval pod can load your submission.)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # REQUIRED for Phi-3; safe + correct for all archs
        **kw,
    )
    return model, tok


def run_experiment(config: MethodConfig) -> dict[str, Any]:
    """Train a model and return its location. See THE CONTRACT above."""
    if config.dry_run:
        # Local smoke path: no training, return a stub ref the toy eval understands.
        return {"model_path": "stub:perfect", "method": "template"}

    # ----------------------------------------------------------------------
    # YOUR METHOD HERE. Typical shape:
    #   model, tok = load_base_model_and_tokenizer(config.base_model)  # eager attn — Phi-3 safe
    #   train_data = load_allowed_corpus(...)        # NOT benchmark items
    #   model = your_training_procedure(model, tok, train_data)   # LoRA SFT, DPO,
    #                                                             # steering, on-policy, ...
    #   model.save_pretrained(config.output_dir); tok.save_pretrained(config.output_dir)
    # ALWAYS load via load_base_model_and_tokenizer (or pass attn_implementation="eager"):
    # the default SDPA/FlashAttn backend CRASHES Phi-3 during on-policy/generation training.
    # And SAVE THE TOKENIZER next to the model — the eval pod must load it to score you.
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Implement your training procedure, save the model to config.output_dir, "
        "and return {'model_path': config.output_dir, 'method': <name>}. "
        "Use dry_run=True for a local smoke test."
    )


if __name__ == "__main__":
    # Local smoke: prove the contract returns a model ref without training.
    out = run_experiment(MethodConfig(dry_run=True))
    print("submission:", out)
    assert set(out) >= {"model_path", "method"}, "method must return model_path + method"
    print("OK: template returns a submit-model result")
