"""Benchmark discovery. Walks benchmarks/* and imports each plugin so its
`__init_subclass__` registers it. Mirrors how safety_aar discovers methods.

CLI:
    python -m aar.benchmarks.registry list
"""
from __future__ import annotations

import importlib
import pkgutil
import sys

from aar import benchmarks as _pkg
from aar.benchmarks.base import all_benchmarks, get  # noqa: F401 (re-export)


def discover() -> dict[str, type]:
    """Import every benchmark submodule so subclasses register, then return them."""
    for mod in pkgutil.walk_packages(_pkg.__path__, prefix=_pkg.__name__ + "."):
        # Skip the framework modules themselves.
        leaf = mod.name.rsplit(".", 1)[-1]
        if leaf in {"base", "registry", "composite"}:
            continue
        # Skip underscore-prefixed leaf modules: authoring/one-off scripts (_baseline, _pilot,
        # _validate, _regen_*, _render, ...) and shared helpers (_mask_src, _injecagent_src, ...).
        # None of them register a benchmark (registrars live in each plugin's benchmark.py; helpers
        # are imported by their consumers), and several load a model / publish datasets AT IMPORT
        # with no __main__ guard, which would needlessly load a 2B model on every discovery.
        if leaf.startswith("_"):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as e:  # a broken plugin shouldn't kill discovery
            print(f"[registry] skip {mod.name}: {e}", file=sys.stderr)
    return all_benchmarks()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        found = discover()
        if not found:
            print("(no benchmarks registered)")
            return
        print("Registered benchmarks:")
        for name, cls in sorted(found.items()):
            print(f"  {name:20s} category={cls.category:10s} ({cls.__module__})")
    else:
        print("usage: python -m aar.benchmarks.registry list")


if __name__ == "__main__":
    main()
