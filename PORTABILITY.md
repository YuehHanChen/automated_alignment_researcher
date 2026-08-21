# Adopting this harness to your infrastructure

The harness is plain Python configured entirely through environment variables (`aar/config.py`), with
no scheduler or cloud assumptions on the evaluation path. To run it on your own setup — a laptop with a
GPU, an internal cluster, Modal, or any container platform — you only wire a few "seams." Nothing here
hardcodes a specific machine, user, or storage path: every default is repo-relative and
env-overridable. Copy `.env.example` to `.env` and fill it in.

## The seams

| Seam | Env var(s) | What it is |
|---|---|---|
| **Transport** | `HARNESS_TRANSPORT` = `fs` \| `s3` | How the research side and the evaluator exchange the model and the scores (`aar/transport.py`). `fs` = a shared filesystem (a mounted volume). `s3` = any S3-compatible object store. |
| **Holdout (eval-only)** | `HOLDOUT_DIR`, `HELDOUT_SCORES_DIR`, `AAR_BENCHMARK_DOCS` | The secret suite + held-out scores + `benchmark_docs/` (which names the held-out). Must be reachable **only** by the evaluator. |
| **Shared handoff** | `SUBMISSIONS_DIR`, `SCORES_DIR` | The research side stages a model in `SUBMISSIONS_DIR`; the evaluator writes the held-out-stripped composite to `SCORES_DIR`. |
| **Judges** | `JUDGE_BACKEND` (`openai`\|`anthropic`\|`local`), `JUDGE_MODEL`, `JUDGE_MODEL_LOCAL`, `OAI_API`, `ANTHROPIC_API_KEY`, `HF_TOKEN` | Per-axis judge (see `REPRODUCE.md` §4). `local` runs a judge model (Qwen2.5-7B) on the GPU; the others call an API. |
| **Compute** | `CUDA_VISIBLE_DEVICES` | `run_eval` uses every visible GPU (benchmarks scored in parallel). Each trained method uses one GPU for ~30 min. |
| **Repo root** | `WORKSPACE_DIR`, `AAR_REPO` | Repo location; used to resolve configs, briefings, and the bundled `benchmark_docs/`. |

## Isolation is a deployment choice, not baked in

A real behavioral gain (not test-set leakage) is only guaranteed if the research process **cannot read
the held-out data**. The harness enforces the *stripping* (the held-out score never enters
`SCORES_DIR`), but keeping `HOLDOUT_DIR` + `benchmark_docs/` out of the research side's reach is up to
how you deploy:

- **Two OS users** (the paper's setup): the evaluator runs as a user that owns `HOLDOUT_DIR` mode-700;
  the research process runs as a different user. See `ISOLATION.md`.
- **Two containers / two Modal functions**: mount the holdout volume only into the evaluator; give the
  research side only the shared handoff volume.
- **Single machine (trusted, e.g. reproducing baselines)**: the file split is enough — just don't point
  the research loop at `HOLDOUT_DIR`.

## Decoupling from the built-in cloud backend (RunPod + S3)

The harness ships with one *optional* cloud path — GPU pods on **RunPod** and model/score transport over
**S3** — used in the original deployment. You do **not** need it to adopt the harness on Modal, an
internal cluster, or a single box.

- The cloud path lives in `aar/infrastructure/{runpod,s3_utils}.py` and
  `aar/web_ui/backend/eval_orchestration.py` (`deploy_pod`), plus the `s3` branch of `aar/transport.py`.
- **Every cloud credential is read from the environment with an empty default** (`config.py`:
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`,
  `RUNPOD_API_KEY`). Nothing cloud-specific activates unless you set them, and none are stored in the repo.
- **To use your own infra:** keep `HARNESS_TRANSPORT=fs` (the default) so the research and eval sides
  exchange models/scores over a shared filesystem or mounted volume, and dispatch the train/eval jobs
  with your own mechanism (Modal functions, your scheduler, or the portable commands below). Replacing
  `deploy_pod` with a one-function call on your platform is the only integration point.
- The portable evaluation path (`scripts/publish_suite.py` → `aar.eval_pod.run_eval`) imports no
  `runpod`/`boto3` at all, so `REPRODUCE.md` runs anywhere with a GPU.

(The `aar/web_ui/frontend/` React dashboard is also optional — the orchestrator/forum server runs without
it; it is not needed for eval or the loop.)

## Recipes

### Reproduce the eval only (no scheduler needed)
Follow `REPRODUCE.md`: `publish_suite.py` (build a suite once) then `aar.eval_pod.run_eval` (score a
model). Both are plain Python; set the env vars above. This is the portable path and needs no cluster.

### Modal
- Build the image from the included `Dockerfile` (base-agnostic — set your CUDA+PyTorch base).
- Put the shared handoff (`SUBMISSIONS_DIR`, `SCORES_DIR`) on one Modal **Volume**; put `HOLDOUT_DIR` +
  `benchmark_docs/` on a **separate** Volume mounted only into the evaluator function.
- Pass `HF_TOKEN` / `OAI_API` / `ANTHROPIC_API_KEY` as Modal **secrets**.
- Wrap two functions: `publish_suite` (run once per axis) and `run_eval` (per submitted model); the
  research loop (`run.py agent`) is a third function. `HARNESS_TRANSPORT=fs` over the shared Volume, or
  `s3` if you prefer object storage.

### Internal cluster / Slurm
- The authors' original Slurm scripts (`scripts/eval_worker.sh`, `eval_job.sh`,
  `launch_eval_worker.sh`, `baseline_*.sh`) are included **as reference** — their paths are placeholders
  (`/opt/aar/...`, `AAR_REPO`); edit them for your site, or just use the portable `REPRODUCE.md`
  commands under your own job wrapper.
- The evaluator is a simple drain loop: it watches `SUBMISSIONS_DIR` for staged models and scores each.
  Point `SUBMISSIONS_DIR` / `SCORES_DIR` / `HOLDOUT_DIR` at your shared and eval-only storage.

## Training backend

Evaluation runs out of the box. The AAR **training** step is executed by each method's own submitted
`run.py` (it may use any trainer — LoRA/PEFT, TRL, etc.); the harness only orchestrates and dispatches
single-GPU jobs. No specific trainer is bundled, so wire your trainer/launcher of choice into the
research loop for full end-to-end runs.

## Keys and safety

- All keys come from the environment; **no key is stored in the repo**. `.env` is gitignored.
- `benchmark_docs/` and `HOLDOUT_DIR` name/contain the held-out — treat them as eval-only.
