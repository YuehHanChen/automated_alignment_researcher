#!/bin/bash
#
# DEPRECATED / DISABLED — this was the single-user "train + eval in one Slurm job
# as the same user" flow. It read a research-readable holdout, which the
# kernel-enforced two-user isolation deliberately removes: the holdout is now
# owned by a separate eval user (mode 700) and the research user cannot read it.
#
# Use the hardened two-user flow instead:
#   1. As the EVAL user (once):   sbatch scripts/eval_worker.sh sycophancy
#   2. As the RESEARCH user:      sbatch scripts/slurm_train_submit.sh <idea> [run_id]
#
# slurm_train_submit.sh trains the method, stages the model to the shared
# submissions channel, and polls for the score the eval worker writes after
# scoring against the locked holdout — exactly what one AAR iteration does, but
# without the research side ever touching the holdout.
#
# This script intentionally exits non-zero so nobody silently re-introduces a
# research-readable holdout. See ISOLATION.md.

echo "ERROR: slurm_run_method.sh is disabled under two-user isolation." >&2
echo "  Use:  sbatch scripts/eval_worker.sh sycophancy        (as the eval user, once)" >&2
echo "  Then: sbatch scripts/slurm_train_submit.sh ${1:-<idea>} ${2:-}" >&2
echo "  See ISOLATION.md." >&2
exit 1
