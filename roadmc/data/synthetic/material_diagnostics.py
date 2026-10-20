"""Integrals of the latent replay loss field; not measured material volume."""

from __future__ import annotations

import numpy as np


def loss_diagnostics(x, y, loss, owner, footprints, event_ids):
    """Trapezoidal integration on node-centred grids, including edge half weights.

    Loss is measured against the current replay reference, which patches reset.
    Texture relief, bulk density, fracture voids and hidden layers are excluded.
    Final ownership avoids counting overlapping events more than once.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    loss, owner = np.asarray(loss), np.asarray(owner)
    footprints = np.asarray(footprints, dtype=bool)
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2:
        raise ValueError("two finite, increasing grid axes required")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("two finite, increasing grid axes required")
    shape = (len(x), len(y))
    if loss.shape != shape or owner.shape != shape or footprints.shape != (len(event_ids), *shape):
        raise ValueError("loss, owner and footprints must align with grid and event IDs")
    if not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError("finite nonnegative loss required")
    if (
        not np.issubdtype(owner.dtype, np.integer)
        or np.any(owner < -1)
        or np.any(owner >= len(event_ids))
    ):
        raise ValueError("owner must be -1 or a valid event index")
    if np.any((loss > 0) & (owner < 0)):
        raise ValueError("positive loss requires an event owner")

    def weights(axis):
        delta = np.diff(axis)
        return np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]

    area = weights(x)[:, None] * weights(y)[None, :]
    active = loss > 0
    volume = float(np.sum(area * loss))
    per_event = []
    for index, ident in enumerate(event_ids):
        mask = active & (owner == index)
        owned_area = float(np.sum(area[mask]))
        owned_volume = float(np.sum(area[mask] * loss[mask]))
        per_event.append(
            {
                "instance_id": ident,
                "footprint_area_m2": float(np.sum(area[footprints[index]])),
                "final_owned_loss_area_m2": owned_area,
                "final_owned_loss_volume_m3": owned_volume,
                "mean_owned_loss_m": owned_volume / owned_area if owned_area else None,
                "max_owned_loss_m": float(loss[mask].max()) if mask.any() else 0.0,
            }
        )
    return {
        "status": "synthetic_latent_loss_only_not_real_measurement",
        "integration": "tensor trapezoidal node weights; indicator areas approximate raster boundaries",
        "reference": "current low-frequency reference; patches reset earlier loss",
        "excludes": "texture relief, density, hidden layers and previously reset loss",
        "surface_area_m2": float(area.sum()),
        "active_loss_area_m2": float(area[active].sum()),
        "loss_volume_m3": volume,
        "max_loss_m": float(loss.max()),
        "events": per_event,
    }
