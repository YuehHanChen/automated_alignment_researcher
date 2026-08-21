"""jbb_artifacts — refusal robustness UNDER ATTACK (JailbreakBench precomputed artifacts).

The "under attack" facet: JBB harmful behaviors wrapped in precomputed adversarial strings
(PAIR / GCG-transfer / random-search), applied to our targets as a TRANSFER setting. Same
joint anti-gaming scorer + paper-faithful judges as `jbb` (Llama-Guard-3-8B harmful arm,
Llama-3-8B refusal judge benign arm). The benign-under-attack arm wraps JBB benign with the
same content-agnostic suffix (GCG / random-search) or the plain benign prompt (PAIR).

Headroom caveat (audit): PAIR/GCG were optimized on Vicuna/Llama-2, so transfer ASR to our
targets may be low — pilot per-target (a non-landing attack must not read as robustness).

Secret data: secret_dir/jbb_artifacts.jsonl, one
{"prompt": <adversarial request>, "harmful": <bool>, "wrapper": <method>} per line.
"""
from __future__ import annotations

from pathlib import Path

from aar.benchmarks.base import Benchmark, Model
from aar.benchmarks.jbb.benchmark import _joint_refusal_score


class JBBArtifacts(Benchmark):
    name = "jbb_artifacts"
    category = "judge"
    judge_model = "meta-llama/Llama-Guard-3-8B"   # harmful arm; benign arm = Llama-3-8B refusal judge

    def score(self, model: Model):
        return _joint_refusal_score(Path(self.secret_dir) / "jbb_artifacts.jsonl", model, self.spec.subset_size)
