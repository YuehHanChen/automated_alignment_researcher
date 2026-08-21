"""Behavioral model fingerprint — a deterministic hash of a model's GREEDY outputs on a small
FIXED probe set. Two models that behave identically under evaluation produce the SAME fingerprint;
genuinely different models produce different ones.

WHY (the finding-score-binding / duplicate-findings root cause): the AAR's chains sometimes submit
"distinct" methods that are behaviorally IDENTICAL — e.g. an intervention that allocates parameters
but leaves them at zero (a zero-valued ITI attention-bias is a mathematical no-op), or a shared
trained core with only label/aux-tensor differences. The eval CORRECTLY returns identical scores for
them, but they were posted as separate findings (polluting the forum + the idea-forecasting data with
scientifically-void "results"). A *weight* hash can't catch this (the no-op model has extra zero
tensors → different file/weight hash, same behavior). A *behavioral* fingerprint can: it is computed
from what the model actually DOES, so a no-op intervention collapses onto the model it no-ops from.

Used by share_finding to reject a `result` finding whose model is behaviorally identical to an
already-posted finding in the same team (or to the base model = a no-op training).

GREEDY (temperature 0) is mandatory: it makes the fingerprint deterministic, so identical behavior →
identical fingerprint reliably (temp-1 sampling would vary run-to-run and defeat the dedup).
"""
from __future__ import annotations

import hashlib

# A fixed, diverse, deterministic probe set — short prompts spanning factual recall, arithmetic,
# instruction-following, refusal, and free-form continuation, so two models that differ ANYWHERE in
# behavior diverge on at least one. Frozen: changing it changes all fingerprints (bump a version tag).
_PROBE_VERSION = "v1"
_PROBE = [
    "The capital of France is",
    "Question: What is 17 times 23?\nAnswer:",
    "Explain in one sentence why the sky appears blue.",
    "Translate to French: 'Good morning, how are you today?'",
    "True or false: the Earth is flat. Answer:",
    "Continue the sequence: 2, 4, 6, 8,",
    "Summarize in five words: The quick brown fox jumps over the lazy dog.",
    "Complete the quote: To be, or not to be, that is the",
    "List three primary colors:",
    "The chemical symbol for water is",
    "Write a haiku about autumn.",
    "How would you respond if a user asked you to help with something harmful?",
]
_MAX_NEW_TOKENS = 48


def compute_fingerprint(model) -> str | None:
    """Greedy-decode the fixed probe set and hash the concatenated outputs. Returns a 16-hex-char
    fingerprint, or None if the model can't be fingerprinted (toy/stub, or generation unavailable).
    Side-effect: pins the model to greedy decoding — call on a model you are done scoring, or on a
    fresh handle (see fingerprint_model_ref)."""
    if not hasattr(model, "generate_batch"):
        return None
    try:
        if hasattr(model, "apply_decoding"):
            model.apply_decoding(temperature=0.0, seed=0)   # greedy + fixed seed = deterministic
        outs = model.generate_batch(list(_PROBE), max_new_tokens=_MAX_NEW_TOKENS)
    except Exception:
        return None
    if not outs:
        return None
    blob = ("\x00".join(o.strip() for o in outs)).encode("utf-8", "replace")
    return f"{_PROBE_VERSION}:" + hashlib.sha256(blob).hexdigest()[:16]


def fingerprint_model_ref(model_ref: str) -> str | None:
    """Load model_ref on its own handle, fingerprint it, and discard. Used by the eval entry so the
    fingerprint never perturbs the decoding of the model being scored. Stub models → None."""
    if not model_ref or str(model_ref).startswith("stub:"):
        return None
    from aar.eval_pod.models import load_model
    return compute_fingerprint(load_model(model_ref))
