"""machiavelli_heldout — the POWER-SEEKING (#11) HELD-OUT generalization probe.

IDENTICAL scorer + construct as `machiavelli_power` (the binary gratuitous-harm contrast, ↑=safer), on a
DISJOINT set of MACHIAVELLI game titles (the games bucketed out of the scored set; see _publish game_split).
Same clean rule scorer, OOD scenarios → an OOD-SCENARIO generalization read (the expert_factor / winobias /
sciq_uncertainty pattern). role=held_out (stripped from the AAR composite + views, scored eval-private).
Caveat: shares the source + scorer with machiavelli_power, so it tests SCENARIO-generalization, not
scorer/format-generalization (a method gaming the MACHIAVELLI choice format moves both).
"""
from __future__ import annotations

from aar.benchmarks.machiavelli_power.benchmark import MachiavelliPower


class MachiavelliHeldout(MachiavelliPower):
    name = "machiavelli_heldout"
