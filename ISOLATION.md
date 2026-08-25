# Isolation — LIVE two-user setup (kernel-enforced)

**Status: enforced.** The secret holdout is owned by a separate OS user
(`eval-user`, uid 2146) at `/opt/aar/eval-user/holdout`,
mode 700. The research user `aar-user` (what the AAR runs as) **cannot read
it** — verified: `ls`/`cat`/`sudo` on the holdout all return Permission denied
(the research user has no *passwordless* sudo, so an automated AAR can't
escalate). Both users share the `shared` group (gid 10000), which carries the
model + scores between them.

## Topology
```
research (aar-user)                         eval (eval-user)
  train method -> stage model                    eval_worker.sh (GPU, looping):
  transport.put_model ->                           reads HOLDOUT_DIR (mode 700, only it can read)
    submissions/<run_id>/model + .submitted        reads submissions/<run_id>/model (shared grp)
  poll scores/<run_id>.json  <----------------     scores -> scores/<run_id>.json (shared grp)
  (CANNOT read the holdout)                         (writes only to the shared scores channel)
```
- `submissions/` and `scores/` live under `/opt/aar/work/aar_repo_runs`,
  group `shared`, setgid `2770` — cross-user reads, neither writes the other's dirs.
- `HOLDOUT_DIR` (eval-only) = `/opt/aar/eval-user/holdout`.

## Operating it
```bash
# 1. EVAL side (secret holder) — PREREQUISITE: publish the holdout for this axis+model
#    BEFORE anything else (see "Publishing the holdout" below). The AAR never sees this step.
ssh -i ~/.ssh/id_ed25519 -p 19934 eval-user@EVAL_HOST \
  'cd /opt/aar/aar_repo && AXIS=sycophancy MODEL=mistral scripts/publish_holdout.sh'

# 2. EVAL side — start the worker for that axis (serves all chains):
ssh ... eval-user@... 'cd .../aar_repo && AXIS=sycophancy scripts/launch_eval_worker.sh'

# 3. RESEARCH side (the AAR / you) — launch the team for the SAME axis+model:
ssh cluster 'cd /opt/aar/work && AXIS=sycophancy MODEL=mistral scripts/launch_team.sh "alpha beta" 100 47'
```
Order matters: **publish_holdout (1) must precede the worker (2) and the AAR (3)** — it is
the eval-side step that makes the run per-model and is invisible to the AAR.
The autonomous chain (`slurm_aar_chain.sh`) does step 2 internally per iteration
(its `evaluate_model` tool stages the model and polls — it does NOT run its own
eval, because `EVAL_VIA_WORKER=true` and it can't read the holdout); keep one
eval worker running to serve all chains.

## What enforces it (and what was removed)
- The **only** copy of the holdout is the eval-owned mode-700 dir above. The old
  research-readable copy at `/opt/aar/work/aar_repo_holdout`
  has been **deleted** — leaving it would have let the AAR read the answer keys
  directly and defeat the lock. (It's regenerable: see re-publish, below.)
- `EVAL_VIA_WORKER` (config.py, default true): when set, the research side's
  `evaluate_model` only **polls** `SCORES_DIR`; it never spawns an eval that
  would need the holdout. Set false only on a legacy single-user dev box.
- `scripts/slurm_run_method.sh` (old single-user "train + eval in one job") is
  **disabled** — it exits non-zero pointing at `slurm_train_submit.sh`, so no one
  re-introduces a research-readable holdout. Use `slurm_train_submit.sh` instead.

## Publishing the holdout — REQUIRED before each (axis, model) run (eval user only)

**`publish_holdout.sh` MUST be run, by the eval user, for the axis+model you're about to
run, BEFORE the AAR is launched.** It (1) writes the secret holdout suite (incl. the
held-out) into the mode-700 dir the worker reads, with that **model's** per-model
baselines, and (2) emits the held-out-stripped per-model prompt baselines the AAR sees.
Without it the worker has no suite to score and the AAR's prompt has no (or wrong-model)
baselines.

```bash
# as the eval user, for the axis + model you'll run:
ssh -i ~/.ssh/id_ed25519 -p 19934 eval-user@EVAL_HOST \
  'cd /opt/aar/aar_repo && \
   AXIS=sycophancy MODEL=mistral scripts/publish_holdout.sh && \
   chmod -R 700 /opt/aar/eval-user/holdout'
```
(`publish_holdout.sh` reads per-model baselines from `benchmark_docs/` via `AAR_BENCHMARK_DOCS`
— deploy that **eval-side mode-700** too, since the per-axis files name the held-out.)

**This is an EVAL-SIDE operator step the AAR never sees.** It is NOT in the AAR's prompt,
not exposed by any tool, and the AAR never runs it — so the AAR is not told that a holdout-
publishing step (or the held-out it sets up) exists. Keep it that way: document the
prerequisite **here / in the eval runbook only**, never in `prompt_safety.jinja2` or any
AAR-facing string. (Verified: the prompt contains no reference to `publish_holdout`,
`benchmark_docs`, or the baselines file.)

## Backup
The holdout answer keys are reproducible (publish script in git). Back up the
small key files to cold storage / a versioned bucket; see below.

---

# (Legacy / superseded) single-user / honor-system notes

> **Historical.** This describes the original single-user posture. The live
> setup above replaced it with the kernel-enforced two-user boundary, and the
> research-readable holdout these notes refer to has been deleted. Kept only to
> explain the design lineage and the S3 alternative.

The harness uses **`HARNESS_TRANSPORT=fs`** by default: the trained model and
scores move over the shared VAST filesystem; the secret benchmark suite lives
in `HOLDOUT_DIR`. The S3 path (`HARNESS_TRANSPORT=s3`) exists for infra with no
shared filesystem (cross-data-center pods) and is otherwise unused on the
Slurm cluster.

## Layout

```
/opt/aar/work
  aar_repo_holdout/<suite>/        # SECRET: test inputs + answers + <suite>.yaml
    <suite>.yaml                      #   (the suite config lives WITH the secret)
    ...                               #   read only by the eval job
  aar_repo_runs/submissions/<run_id>/
    model/                            # the AAR's submitted model (ephemeral, regenerable)
    scores.json                       # aggregate scores (the only thing the AAR gets back)
```

`HOLDOUT_DIR` is deliberately **outside** any AAR worktree / results dir, so the
AAR's routine `rm -rf` cleanups can never reach it.

## Isolation: dev vs production

- **Dev (single OS user — today):** honor system. The AAR is *told* (in its
  prompt / SKILL rules) not to read `HOLDOUT_DIR`, but nothing enforces it
  because the AAR and the eval run as the same user (`$USER`). This is the same
  posture our `safety_aar` eval server had in dev.

- **Production hardening (one admin ask):** have Eugene create a separate
  **eval user** (or group) that owns `HOLDOUT_DIR`:
  ```bash
  # as the eval user:
  chown -R evaluser:evalgroup /opt/aar/work
  chmod -R 700 /opt/aar/work
  ```
  Then run `scripts/slurm_eval.sh` as the eval user (`sbatch --uid` / a
  submit-as-eval wrapper, or the eval user owns the cron that drains the
  submissions queue). Now the AAR (running as `$USER`) physically **cannot read
  or delete** the holdout — the same kernel boundary enforces both
  confidentiality and deletion-safety. This is the filesystem analog of the
  S3 credential split, and a lighter admin ask than a second IAM principal.

## Data safety / backup

VAST is persistent (survives restarts; not reaped like dev pods or `/tmp`). The
real risks are an accidental `rm` and quota pressure. Mitigations:

1. **Separate-user mode-700 (above)** also blocks accidental deletion by the AAR.
2. **The holdout is reproducible** — keep the publish script (that builds
   `aar_repo_holdout/<suite>/` from public datasets + your secret answer
   keys) in git. Regenerate anytime.
3. **Back up the answer keys** to durable storage:
   - the cluster's slow **network cold-storage drive** (handbook's intended use), or
   - a **versioned GCS/S3 bucket** (object versioning makes even a catastrophic
     delete recoverable), or
   - a **private git repo**.
   Back up only the small answer-key files, not the regenerable bulk.

## Switching to S3 later

Set `HARNESS_TRANSPORT=s3` + `S3_BUCKET`/creds/`EVAL_POD_TEMPLATE_ID`. The model
then round-trips object storage and the eval runs on an ephemeral RunPod pod.
Only worth it if the eval can't share a filesystem with the research side.
