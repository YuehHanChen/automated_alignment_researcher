# scripts/models.sh — the target models for the AAR sweep. `source` this (don't execute).
#
# The model is INDEPENDENT of the safety axis: you pick an AXIS (what to optimize) and a
# MODEL (which model to optimize) separately. axis_env.sh sources this and resolves
# MODEL -> TARGET_MODEL. Use a short alias or the full HF id; anything else fails fast
# (so a typo can't silently run the wrong / an unintended model).
#
#   resolve_model <alias|full-hf-id>  -> echoes the full HF id; returns 1 if not one of the set.
#
# The 5 NEXT-GEN target models (the temp-1 baselines + dont_run + benchmark_docs are all built
# for THIS set; the prior previous-gen set — Qwen2.5-3B, Mistral-7B, OLMo-2-7B, Phi-3.5 — is retired):
#   qwen     Qwen/Qwen3.5-2B
#   llama    meta-llama/Llama-3.2-3B-Instruct
#   olmo     allenai/Olmo-3-7B-Instruct
#   gemma    google/gemma-2-2b-it
#   phi      microsoft/Phi-4-mini-instruct
MODELS_ALIASES="qwen llama olmo gemma phi"

resolve_model() {
  # Accept either the alias or the exact full HF id (so TARGET_MODEL=<full id> still works).
  case "${1:-}" in
    qwen|Qwen/Qwen3.5-2B)                   echo "Qwen/Qwen3.5-2B" ;;
    llama|meta-llama/Llama-3.2-3B-Instruct) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
    olmo|allenai/Olmo-3-7B-Instruct)        echo "allenai/Olmo-3-7B-Instruct" ;;
    gemma|google/gemma-2-2b-it)             echo "google/gemma-2-2b-it" ;;
    phi|microsoft/Phi-4-mini-instruct)      echo "microsoft/Phi-4-mini-instruct" ;;
    *) return 1 ;;
  esac
}
