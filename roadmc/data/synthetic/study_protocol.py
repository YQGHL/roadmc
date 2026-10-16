"""Admission and provenance for bounded procedural-realism studies.

Unknown units and unknown specimen identities remain explicit. A successful
software run never promotes a reference into calibrated real-road evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_object,
        parse_constant=_constant,
    )


def canonical_hash(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement only of this study's own ledger/result file.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


ABSOLUTE_METRICS = {"height_rms_m", "depth_m", "volume_m3", "slope_metric"}
PLANAR_METRICS = {"width_m", "area_m2", "psd_cycles_m"}
TEXTURE_ASSETS = {"texture_archive", "texture_split", "texture_prior"}
NATIVE_ASSETS = {
    f"pothole_{kind}_{frame}" for kind in ("frame", "label") for frame in ("01", "02", "03")
}
OUTLINE_FAIRNESS_ASSETS = NATIVE_ASSETS | {"pothole_frozen_metrics"}


def _validate_study_import(source, binding, paths, owners):
    """Keep the study declaration within a hashed native import's evidence."""
    ident = source["source_id"]
    if not isinstance(binding, dict) or set(binding) != {"npz_asset", "provenance_asset"}:
        raise ValueError(f"{ident}: study_import requires npz_asset and provenance_asset")
    names = (binding["npz_asset"], binding["provenance_asset"])
    if any(
        not isinstance(name, str) or name not in paths or owners[name] != ident for name in names
    ):
        raise ValueError(f"{ident}: study_import assets must bind this admitted source")
    npz_path, provenance_path = (Path(paths[name]) for name in names)
    provenance = read_json(provenance_path)
    if not isinstance(provenance, dict) or provenance.get("format") != "roadmc.study-import.v1":
        raise ValueError(f"{ident}: unsupported study import provenance")
    for key in (
        "source_id",
        "material_id",
        "exposure",
        "role",
        "units",
        "unit_evidence",
        "processing_chain",
    ):
        if key not in source or source[key] != provenance.get(key):
            raise ValueError(f"{ident}: study import {key} declaration differs from provenance")
    if source.get("group_identity") != provenance.get("group_status") or source.get(
        "group_id"
    ) != provenance.get("group_id"):
        raise ValueError(f"{ident}: study import group declaration differs from provenance")
    if not set(source["allowed_metrics"]).issubset(provenance.get("allowed_metrics", [])):
        raise ValueError(f"{ident}: study import cannot expand allowed_metrics")
    if provenance.get("npz") != npz_path.name or provenance.get("npz_sha256") != file_hash(
        npz_path
    ):
        raise ValueError(f"{ident}: study import provenance does not match NPZ hash/name")
    # This format is intentionally native-only. A hand-edited flag cannot upgrade
    # it into a metric RawRoadScene or an independent-road validation bundle.
    if any(
        provenance.get(key) is not False
        for key in ("independent_validation_eligible", "metric_export_eligible")
    ):
        raise ValueError(f"{ident}: native import v1 eligibility flags must remain false")
    if source["role"] == "independent_holdout":
        raise ValueError(f"{ident}: native import is not eligible for independent holdout")
    if (ABSOLUTE_METRICS | PLANAR_METRICS).intersection(source["allowed_metrics"]):
        raise ValueError(f"{ident}: native import is not eligible for metric export")
    return {
        "format": provenance["format"],
        "npz_asset": names[0],
        "provenance_asset": names[1],
        "npz_sha256": provenance["npz_sha256"],
        "provenance_sha256": file_hash(provenance_path),
        "metric_export_eligible": provenance.get("metric_export_eligible") is True,
        "independent_validation_eligible": provenance.get("independent_validation_eligible")
        is True,
    }


def validate_manifest(manifest, base_path):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("study manifest must have schema_version=1")
    datasets = manifest.get("datasets")
    assets = manifest.get("assets")
    if not isinstance(datasets, list) or not datasets or not isinstance(assets, dict):
        raise ValueError("manifest needs datasets array and assets mapping")
    ids = set()
    groups = {}
    audit = []
    for source in datasets:
        if not isinstance(source, dict):
            raise ValueError("dataset rows must be objects")
        ident = source.get("source_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("source_id must be a unique nonempty string")
        ids.add(ident)
        units = source.get("units", {})
        if (
            not isinstance(units, dict)
            or units.get("xy") not in {"m", "mm", "um", "native", "pixel", "unknown"}
            or units.get("z") not in {"m", "mm", "um", "native", "gray", "unknown"}
        ):
            raise ValueError(f"{ident}: explicit xy/z unit labels required")
        unit_evidence = source.get("unit_evidence", {})
        if not isinstance(unit_evidence, dict):
            raise ValueError(f"{ident}: unit_evidence must be an object")
        metrics = source.get("allowed_metrics")
        if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
            raise ValueError(f"{ident}: explicit allowed_metrics required")
        xy_verified = units["xy"] in {"m", "mm", "um"} and unit_evidence.get("xy_verified") is True
        z_verified = units["z"] in {"m", "mm", "um"} and unit_evidence.get("z_verified") is True
        if ABSOLUTE_METRICS.intersection(metrics) and not (xy_verified and z_verified):
            raise ValueError(f"{ident}: absolute geometry metrics require verified XY and Z units")
        if PLANAR_METRICS.intersection(metrics) and not xy_verified:
            raise ValueError(f"{ident}: metric planar descriptors require verified XY units")
        if source.get(
            "reference_kind"
        ) == "range_image_unknown_z" and ABSOLUTE_METRICS.intersection(metrics):
            raise ValueError(f"{ident}: image gray values are not metric elevations")
        identity = source.get("group_identity")
        if identity not in {"verified", "unknown", "possibly_related"}:
            raise ValueError(f"{ident}: explicit group_identity required")
        role = source.get("role")
        if role not in {
            "fit",
            "seen_benchmark",
            "diagnostic",
            "candidate_only",
            "independent_holdout",
        }:
            raise ValueError(f"{ident}: unsupported role")
        exposure = source.get("exposure")
        if exposure not in {"seen", "unseen", "unknown"}:
            raise ValueError(f"{ident}: explicit exposure required")
        if role == "independent_holdout" and (
            identity != "verified"
            or exposure != "unseen"
            or source.get("material_id") not in {"asphalt", "concrete"}
        ):
            raise ValueError(
                f"{ident}: independent holdout requires verified identity/material and unseen exposure"
            )
        group = source.get("group_id")
        if identity == "verified":
            if not isinstance(group, str) or not group:
                raise ValueError(f"{ident}: verified groups need group_id")
            group_roles = groups.setdefault(group, set())
            group_roles.add(role)
            if "independent_holdout" in group_roles and len(group_roles) > 1:
                raise ValueError(f"{ident}: one group cannot enter fitting and independent holdout")
        audit.append(
            {
                "source_id": ident,
                "material_id": source.get("material_id"),
                "exposure": exposure,
                "role": role,
                "allowed_metrics": metrics,
                "metric_xy_verified": xy_verified,
                "metric_z_verified": z_verified,
                "independent_identity_verified": identity == "verified",
                "real_domain_accepted": False,
            }
        )
    resolved, asset_sources = {}, {}
    for name, asset in assets.items():
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise ValueError(f"asset {name}: path required")
        if asset.get("source_id") not in ids:
            raise ValueError(f"asset {name}: source_id must bind an admitted dataset")
        path = Path(asset["path"])
        if not path.is_absolute():
            path = Path(base_path) / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"asset {name}: missing file {path}")
        expected = asset.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
            raise ValueError(f"asset {name}: SHA256 mismatch or missing declared hash")
        resolved[name] = str(path)
        asset_sources[name] = asset["source_id"]
    for source, row in zip(datasets, audit, strict=True):
        if "study_import" in source:
            row["study_import"] = _validate_study_import(
                source, source["study_import"], resolved, asset_sources
            )
    return {
        "datasets": audit,
        "resolved_assets": resolved,
        "asset_sources": asset_sources,
        "calibration_status": "not_calibrated",
        "real_domain_status": "not_evaluated",
        "note": "Admission restricts allowed conclusions; it does not certify instruments or roads.",
    }


def validate_stage_assets(stage, admission):
    """Bind every real file used by an experiment to its admitted purpose."""
    required = (
        TEXTURE_ASSETS
        if stage.startswith("texture_")
        else OUTLINE_FAIRNESS_ASSETS
        if stage == "pothole_outline_fairness"
        else NATIVE_ASSETS
        if stage == "pothole_native"
        else set()
    )
    paths = admission["resolved_assets"]
    if not required.issubset(paths):
        raise ValueError(f"{stage}: missing required assets {sorted(required - paths.keys())}")
    if not required:
        return
    owners = {admission["asset_sources"][name] for name in required}
    if len(owners) != 1:
        raise ValueError(f"{stage}: all required assets must bind one admitted source")
    source = next(row for row in admission["datasets"] if row["source_id"] in owners)
    allowed = set(source["allowed_metrics"])
    if stage.startswith("texture_"):
        if (
            source["role"] != "fit"
            or source["material_id"] != "asphalt"
            or not source["metric_xy_verified"]
            or not {"psd_cycles_m", "normalized_acf", "standardized_height_w1"}.issubset(allowed)
        ):
            raise ValueError(
                f"{stage}: source not admitted for train-only normalized asphalt texture study"
            )
    else:
        needed = {"native_outline_relative_residual"}
        if stage == "pothole_native":
            needed.add("train_q90_normalized_depth")
        if source["role"] != "diagnostic" or not needed.issubset(allowed):
            raise ValueError(f"{stage}: source not admitted for native-unit diagnostics")
        root = Path(paths["pothole_frame_01"]).parent.parent
        for frame in ("01", "02", "03"):
            for kind, folder, suffix in (("frame", "ptcloud", ".mat"), ("label", "label", ".png")):
                if (
                    Path(paths[f"pothole_{kind}_{frame}"])
                    != (root / folder / (frame + suffix)).resolve()
                ):
                    raise ValueError(
                        f"{stage}: all six frozen files must match the actual dataset read paths"
                    )


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("study protocol must have schema_version=1")
    texture = protocol.get("texture")
    if (
        not isinstance(texture, dict)
        or texture.get("selection") != "train_only_loocv_pareto_noninferior"
    ):
        raise ValueError("texture selection must be frozen train-only LOOCV/Pareto rule")
    seeds = texture.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if (
        texture.get("multipliers") != [0.75, 1.0, 1.25]
        or texture.get("iterations") != 50
        or texture.get("tolerance") != 1e-5
    ):
        raise ValueError("protocol changes the approved fixed numerical search")
    if protocol.get("promote_production_defaults") is not False:
        raise ValueError("this study does not automatically promote production defaults")
    return protocol
