"""Importing this package registers the example benchmarks (each subclass auto-registers via
`__init_subclass__` in aar.benchmarks.base). To add your own task: drop a module in this folder
(copy `_TEMPLATE.py`) and import it below so it registers before eval.

`_TEMPLATE.py` is intentionally NOT imported here — it is a copy-me scaffold, not a live benchmark.
"""
from generic_aar.benchmarks import example_hillclimb   # noqa: F401  (gen_task_main)
from generic_aar.benchmarks import example_judge        # noqa: F401  (gen_task_judge)
from generic_aar.benchmarks import example_heldout      # noqa: F401  (gen_task_ood, held-out)
from generic_aar.benchmarks import example_capability   # noqa: F401  (gen_capability, gate)
