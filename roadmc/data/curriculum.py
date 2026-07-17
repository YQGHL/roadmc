"""Label spaces for the RoadMC binary-to-38-class curriculum.

The mappings are partitions of the original JTG labels: a curriculum stage
changes task granularity, never discards or duplicates a source label.
"""

from __future__ import annotations


LABEL_STAGES: tuple[str, ...] = ("binary", "four", "eight", "full38")


def _build_lut(groups: dict[int, tuple[int, ...]], num_classes: int) -> tuple[int, ...]:
    lut = [-1] * 38
    for target_class, source_labels in groups.items():
        if not 0 <= target_class < num_classes:
            raise ValueError(f"invalid target class {target_class}")
        for source_label in source_labels:
            if not 0 <= source_label < 38 or lut[source_label] != -1:
                raise ValueError(f"invalid or duplicate source label {source_label}")
            lut[source_label] = target_class
    if any(value < 0 for value in lut):
        raise ValueError("curriculum mapping must cover every original label")
    return tuple(lut)


_BINARY_LUT = _build_lut(
    {0: (0,), 1: tuple(range(1, 38))},
    num_classes=2,
)

# 0 background; 1 cracks; 2 localized surface/material damage;
# 3 broad deformation and repaired surface.
_FOUR_LUT = _build_lut(
    {
        0: (0,),
        1: tuple(range(1, 9)) + (23, 24),
        2: (9, 10, 11, 12, 19, 21, 22, 25, 26, 29, 30, 31, 32, 33, 34, 36),
        3: (13, 14, 15, 16, 17, 18, 20, 27, 28, 35, 37),
    },
    num_classes=4,
)

# Asphalt classes are separated by mechanism; concrete classes retain the
# fracture / joint / surface-repair distinction used by the generator.
_EIGHT_LUT = _build_lut(
    {
        0: (0,),
        1: tuple(range(1, 9)),
        2: (9, 10, 11, 12),
        3: (13, 14, 15, 16, 17, 18),
        4: (19, 20),
        5: tuple(range(21, 27)),
        6: tuple(range(27, 34)),
        7: tuple(range(34, 38)),
    },
    num_classes=8,
)

_FULL38_LUT = tuple(range(38))

LABEL_LUTS: dict[str, tuple[int, ...]] = {
    "binary": _BINARY_LUT,
    "four": _FOUR_LUT,
    "eight": _EIGHT_LUT,
    "full38": _FULL38_LUT,
}

CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "binary": ("Background", "Disease"),
    "four": (
        "Background",
        "Cracking",
        "Localized surface damage",
        "Deformation and repair",
    ),
    "eight": (
        "Background",
        "Asphalt cracking",
        "Asphalt material loss",
        "Asphalt deformation",
        "Asphalt treatment and patching",
        "Concrete fracture",
        "Concrete joint and edge damage",
        "Concrete surface and patching",
    ),
    "full38": tuple(f"Class {label}" for label in range(38)),
}


def normalize_label_stage(stage: str) -> str:
    """Validate and normalize a curriculum stage name."""

    normalized = stage.lower().strip()
    if normalized not in LABEL_LUTS:
        raise ValueError(f"label stage must be one of {LABEL_STAGES}, got {stage!r}")
    return normalized


def label_lut(stage: str) -> tuple[int, ...]:
    """Return a 38-entry lookup table for a curriculum stage."""

    return LABEL_LUTS[normalize_label_stage(stage)]


def num_classes_for_stage(stage: str) -> int:
    """Return the number of output classes for a curriculum stage."""

    return len(CLASS_NAMES[normalize_label_stage(stage)])


def class_names_for_stage(stage: str) -> tuple[str, ...]:
    """Return ordered class names for reporting."""

    return CLASS_NAMES[normalize_label_stage(stage)]


def stage_for_num_classes(num_classes: int) -> str:
    """Infer a stage only when the model output count is unambiguous."""

    matches = [stage for stage in LABEL_STAGES if num_classes_for_stage(stage) == num_classes]
    if len(matches) != 1:
        raise ValueError(f"No unique curriculum stage has {num_classes} classes")
    return matches[0]
