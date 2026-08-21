"""machiavelli_harm — direct-harm/violence facet of POWER-SEEKING (#11), a scored leg.

IDENTICAL scorer + construct as `machiavelli_power` (binary gratuitous-harm contrast, ↑=safer), but the harm
count is restricted to the DIRECT-HARM / VIOLENCE morality labels (killing / physical_harm / non_physical_harm
/ intending_harm) — the "willingness to hurt others" facet, distinct from machiavelli_power's MACHIAVELLIAN-
ACQUISITION facet (manipulation/deception/stealing/spying/trespassing = illegitimate covert advantage). Both
are equal-reward gratuitous-harm contrasts → degenerate-proof; both from MACHIAVELLI (shared source/scorer
family — facet-distinct, NOT item-independent; the reward-hacking School-by-scorer / loophole-by-facet
precedent). Uses the SAME train game-split as machiavelli_power (disjoint from machiavelli_heldout).
"""
from __future__ import annotations

from aar.benchmarks.machiavelli_power.benchmark import MachiavelliPower


class MachiavelliHarm(MachiavelliPower):
    name = "machiavelli_harm"
