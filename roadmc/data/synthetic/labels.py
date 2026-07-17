"""Deterministic label specifications for controlled RoadMC synthesis.

The random generator is useful for diversity, but it cannot prove that a rare
class is present in a dataset. This module makes every non-background JTG
label reachable through one explicit physical generation path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetLabelSpec:
    """Physical recipe used to force one target label into a scene."""

    pavement_type: str
    disease_type: str
    severity: str
    variant: str | None = None


ASPHALT_TARGET_SPECS: dict[int, TargetLabelSpec] = {
    1: TargetLabelSpec("asphalt", "crack", "light", "alligator"),
    2: TargetLabelSpec("asphalt", "crack", "severe", "alligator"),
    3: TargetLabelSpec("asphalt", "crack", "light", "block"),
    4: TargetLabelSpec("asphalt", "crack", "severe", "block"),
    5: TargetLabelSpec("asphalt", "crack", "light", "longitudinal"),
    6: TargetLabelSpec("asphalt", "crack", "severe", "longitudinal"),
    7: TargetLabelSpec("asphalt", "crack", "light", "transverse"),
    8: TargetLabelSpec("asphalt", "crack", "severe", "transverse"),
    9: TargetLabelSpec("asphalt", "pothole", "light"),
    10: TargetLabelSpec("asphalt", "pothole", "severe"),
    11: TargetLabelSpec("asphalt", "raveling", "light"),
    12: TargetLabelSpec("asphalt", "raveling", "severe"),
    13: TargetLabelSpec("asphalt", "depression", "light"),
    14: TargetLabelSpec("asphalt", "depression", "severe"),
    15: TargetLabelSpec("asphalt", "rutting", "light"),
    16: TargetLabelSpec("asphalt", "rutting", "severe"),
    17: TargetLabelSpec("asphalt", "corrugation", "light"),
    18: TargetLabelSpec("asphalt", "corrugation", "severe"),
    19: TargetLabelSpec("asphalt", "bleeding", "light"),
    20: TargetLabelSpec("asphalt", "patching", "-"),
}

CONCRETE_TARGET_SPECS: dict[int, TargetLabelSpec] = {
    21: TargetLabelSpec("concrete", "concrete_damage", "light", "slab_shatter"),
    22: TargetLabelSpec("concrete", "concrete_damage", "severe", "slab_shatter"),
    23: TargetLabelSpec("concrete", "concrete_damage", "light", "slab_crack"),
    24: TargetLabelSpec("concrete", "concrete_damage", "severe", "slab_crack"),
    25: TargetLabelSpec("concrete", "concrete_damage", "light", "corner_break"),
    26: TargetLabelSpec("concrete", "concrete_damage", "severe", "corner_break"),
    27: TargetLabelSpec("concrete", "concrete_damage", "light", "faulting"),
    28: TargetLabelSpec("concrete", "concrete_damage", "severe", "faulting"),
    29: TargetLabelSpec("concrete", "concrete_damage", "-", "pumping"),
    30: TargetLabelSpec("concrete", "concrete_damage", "light", "edge_spall"),
    31: TargetLabelSpec("concrete", "concrete_damage", "severe", "edge_spall"),
    32: TargetLabelSpec("concrete", "concrete_damage", "light", "joint_damage"),
    33: TargetLabelSpec("concrete", "concrete_damage", "severe", "joint_damage"),
    34: TargetLabelSpec("concrete", "concrete_damage", "-", "pitting"),
    35: TargetLabelSpec("concrete", "concrete_damage", "-", "blowup"),
    36: TargetLabelSpec("concrete", "concrete_damage", "-", "exposed_aggregate"),
    37: TargetLabelSpec("concrete", "concrete_damage", "-", "patching"),
}

TARGET_LABEL_SPECS: dict[int, TargetLabelSpec] = {
    **ASPHALT_TARGET_SPECS,
    **CONCRETE_TARGET_SPECS,
}

ALL_DISEASE_LABELS: tuple[int, ...] = tuple(TARGET_LABEL_SPECS)


def target_spec_for_label(label: int) -> TargetLabelSpec:
    """Return the deterministic physical recipe for a non-background label."""

    try:
        return TARGET_LABEL_SPECS[label]
    except KeyError as exc:
        raise ValueError(
            f"target_label must be one of {ALL_DISEASE_LABELS}, got {label}"
        ) from exc
