"""Evidence-bound, native study imports that remain separate from RawRoadScene."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

FORMAT = "roadmc.study-import.v1"
_UNITS = {"m", "mm", "um", "native", "pixel", "gray", "unknown"}
_ROLES = {"fit", "seen_benchmark", "diagnostic", "candidate_only", "independent_holdout"}
_EXPOSURES = {"seen", "unseen", "unknown"}
_GROUPS = {"verified", "unknown", "possibly_related"}
_ABSOLUTE = {"height_rms_m", "depth_m", "volume_m3", "slope_metric"}
_PLANAR = {"width_m", "area_m2", "psd_cycles_m"}


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _implementation_receipt() -> dict[str, dict[str, str]]:
    repo = Path(__file__).resolve().parents[3]
    paths = {
        "study_import.py": Path(__file__).resolve(),
        "import_study_data.py": repo / "roadmc" / "scripts" / "import_study_data.py",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"import implementation source missing: {path}")
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


@dataclass(frozen=True)
class PreparedSource:
    row: dict[str, Any]
    path: Path
    member_bytes: bytes | None
    source_hash: str
    member_hash: str | None


@dataclass(frozen=True)
class PreparedImport:
    manifest_path: Path
    manifest: dict[str, Any]
    sources: tuple[PreparedSource, ...]


def validate_import_manifest(path: str | Path) -> PreparedImport:
    """Validate all declarations and source/member hashes before any output is made."""
    manifest_path = Path(path).resolve()
    manifest = read_manifest(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("import manifest requires schema_version=1")
    rows = manifest.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("import manifest requires nonempty sources array")
    seen: set[str] = set()
    group_roles: dict[str, set[str]] = {}
    prepared: list[PreparedSource] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each source must be an object")
        sid = row.get("source_id")
        if (
            not isinstance(sid, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", sid)
            or sid.endswith(".")
            or sid.lower().split(".")[0]
            in {
                "con",
                "prn",
                "aux",
                "nul",
                *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10)),
            }
            or sid.casefold() in seen
        ):
            raise ValueError("source_id must be a unique safe filename basename")
        seen.add(sid.casefold())
        for field, allowed in (
            ("group_status", _GROUPS),
            ("exposure", _EXPOSURES),
            ("role", _ROLES),
        ):
            if row.get(field) not in allowed:
                raise ValueError(f"{sid}: explicit valid {field} required")
        if row.get("material_id") not in {"asphalt", "concrete", "unknown"}:
            raise ValueError(f"{sid}: material_id must be asphalt, concrete, or unknown")
        group = row.get("group_id")
        if row["group_status"] == "verified" and (
            not isinstance(group, str) or not group.strip() or row["material_id"] == "unknown"
        ):
            raise ValueError(f"{sid}: verified group requires group_id and known material")
        if row["group_status"] == "verified":
            group_roles.setdefault(group, set()).add(row["role"])
        units = row.get("units")
        if (
            not isinstance(units, dict)
            or units.get("xy") not in _UNITS
            or units.get("z") not in _UNITS
        ):
            raise ValueError(f"{sid}: explicit XY and Z units required")
        evidence = row.get("unit_evidence")
        if (
            not isinstance(evidence, dict)
            or not all(k in evidence for k in ("xy_verified", "xy_basis", "z_verified", "z_basis"))
            or any(not isinstance(evidence[k], bool) for k in ("xy_verified", "z_verified"))
            or any(
                not isinstance(evidence[k], str) or not evidence[k].strip()
                for k in ("xy_basis", "z_basis")
            )
        ):
            raise ValueError(
                f"{sid}: explicit XY/Z unit evidence flags and basis statements required"
            )
        chain = row.get("processing_chain")
        if not isinstance(chain, dict) or not chain:
            raise ValueError(f"{sid}: nonempty processing_chain declaration required")
        metrics = row.get("allowed_metrics")
        if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
            raise ValueError(f"{sid}: allowed_metrics array required")
        if _ABSOLUTE.intersection(metrics) or _PLANAR.intersection(metrics):
            raise ValueError(
                f"{sid}: study-import never grants metric exports; use native/normalized metrics only"
            )
        if row.get("group_status") != "verified" and row.get("role") == "independent_holdout":
            raise ValueError(f"{sid}: unknown/related groups cannot be independent holdout")
        if row.get("exposure") != "unseen" and row.get("role") == "independent_holdout":
            raise ValueError(f"{sid}: independent holdout must be unseen")
        source_path = Path(row.get("path", ""))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise ValueError(f"{sid}: source file missing: {source_path}")
        expected = row.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or sha256_file(source_path) != expected.lower()
        ):
            raise ValueError(f"{sid}: source SHA256 mismatch or missing")
        kind = row.get("format")
        member_bytes = None
        member_hash = None
        if kind in {"royal_csv_zip", "sur_zip"}:
            member = row.get("member")
            if (
                not isinstance(member, str)
                or not member
                or ".." in PurePosixPath(member).parts
                or "\\" in member
            ):
                raise ValueError(f"{sid}: safe explicit ZIP member path required")
            member_expected = row.get("member_sha256")
            if not isinstance(member_expected, str) or len(member_expected) != 64:
                raise ValueError(f"{sid}: member_sha256 required")
            with zipfile.ZipFile(source_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError(f"{sid}: ZIP contains duplicate member names")
                if member not in names or archive.getinfo(member).is_dir():
                    raise ValueError(f"{sid}: explicit ZIP member missing or not a file")
                member_bytes = archive.read(member)
            member_hash = sha256_bytes(member_bytes)
            if member_hash != member_expected.lower():
                raise ValueError(f"{sid}: member SHA256 mismatch")
        elif kind == "sur":
            if source_path.stat().st_size < 512:
                raise ValueError(f"{sid}: SUR shorter than 512-byte header")
        elif kind == "field_txt":
            if row.get("column_schema") not in {"xyz_unknown_aux", "profile"}:
                raise ValueError(f"{sid}: explicit field TXT column_schema required")
            if row["column_schema"] == "profile":
                sampling = row.get("xy_sampling")
                if not isinstance(sampling, dict) or not all(
                    k in sampling for k in ("start", "step", "unit")
                ):
                    raise ValueError(f"{sid}: profile requires explicit XY sampling declaration")
        else:
            raise ValueError(f"{sid}: unsupported format {kind!r}")
        prepared.append(
            PreparedSource(row, source_path, member_bytes, expected.lower(), member_hash)
        )
    for group, roles in group_roles.items():
        if "independent_holdout" in roles and roles.intersection({"fit", "seen_benchmark"}):
            raise ValueError(
                f"verified group {group}: cannot mix fitting/seen benchmark and independent holdout"
            )
    return PreparedImport(manifest_path, manifest, tuple(prepared))


def _load_csv_matrix(data: bytes, source_id: str) -> np.ndarray:
    try:
        matrix = np.loadtxt(data.splitlines(), delimiter=",", dtype=np.float64, ndmin=2)
    except Exception as exc:
        raise ValueError(f"{source_id}: invalid numeric CSV matrix: {exc}") from exc
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError(f"{source_id}: CSV matrix must be nonempty and finite")
    return matrix


def _read_sur(source: PreparedSource) -> dict[str, np.ndarray | dict[str, Any]]:
    raw = source.member_bytes if source.member_bytes is not None else source.path.read_bytes()
    h = raw[:512]
    if h[:12].rstrip(b"\0 ") != b"DIGITAL SURF":
        raise ValueError(f"{source.row['source_id']}: invalid SUR magic")
    # Header offsets are tied to the audited third-party reader layout.
    fmt, objects, version, studiable = struct.unpack_from("<hhhh", h, 12)
    non_measured = struct.unpack_from("<h", h, 86)[0]
    bits = struct.unpack_from("<h", h, 98)[0]
    min_point, max_point, nx, ny = struct.unpack_from("<iiii", h, 100)
    count = struct.unpack_from("<i", h, 116)[0]
    spacing = struct.unpack_from("<fff", h, 120)
    step_units = [h[o : o + 16].rstrip(b"\0 ").decode("latin1") for o in (180, 196, 212)]
    units = [h[o : o + 16].rstrip(b"\0 ").decode("latin1") for o in (228, 244, 260)]
    ratios = struct.unpack_from("<fff", h, 276)
    offsets = struct.unpack_from("<fff", h, 466)
    comment_length, private_length = struct.unpack_from("<hh", h, 334)
    if non_measured not in {0, 1}:
        raise ValueError(f"{source.row['source_id']}: unsupported SUR non_measured_points flag")
    if comment_length < 0 or private_length < 0:
        raise ValueError(f"{source.row['source_id']}: negative SUR comment/private length")
    if min_point < -32768 or max_point > 32767 or min_point > max_point:
        raise ValueError(f"{source.row['source_id']}: SUR min/max outside signed 16-bit code range")
    payload_offset = 512 + comment_length + private_length
    canonical_unit = {"m": "m", "mm": "mm", "um": "um", "µm": "um", "μm": "um"}
    declared_units = source.row["units"]
    header_xy = [canonical_unit.get(unit) for unit in units[:2]]
    header_z = canonical_unit.get(units[2])
    if declared_units["xy"] in {"m", "mm", "um"} and any(
        unit is not None and unit != declared_units["xy"] for unit in header_xy
    ):
        raise ValueError(f"{source.row['source_id']}: declared XY unit conflicts with SUR header")
    if (
        declared_units["z"] in {"m", "mm", "um"}
        and header_z is not None
        and header_z != declared_units["z"]
    ):
        raise ValueError(f"{source.row['source_id']}: declared Z unit conflicts with SUR header")
    if (
        (fmt, objects, version, studiable) != (0, 1, 1, 2)
        or bits != 16
        or nx <= 0
        or ny <= 0
        or count != nx * ny
        or payload_offset + count * 2 != len(raw)
    ):
        raise ValueError(
            f"{source.row['source_id']}: unsupported/inconsistent SUR dimensions or bit depth"
        )
    if not all(math.isfinite(v) and v > 0 for v in spacing) or not all(
        math.isfinite(v) for v in (*ratios, *offsets)
    ):
        raise ValueError(f"{source.row['source_id']}: invalid SUR header spacing/ratio/offset")
    codes = np.frombuffer(raw, dtype="<i2", offset=payload_offset).astype(np.int32).reshape(ny, nx)
    valid = np.ones(codes.shape, dtype=bool)
    missing_code = min_point - 2 if non_measured == 1 else None
    if missing_code is not None:
        valid &= codes != missing_code
    masks = {
        "all_codes": np.ones(codes.shape, dtype=bool),
        "exclude_zero": codes != 0,
        "exclude_max_code": codes != max_point,
        "exclude_zero_and_max_code": (codes != 0) & (codes != max_point),
    }
    # Header unit text is preserved as evidence; do not infer instrument accuracy.
    z_header_scaled = codes.astype(np.float64) * spacing[2] * ratios[2] + offsets[2]
    return {
        "values_raw": codes,
        "validity_mask": valid,
        "sensitivity_masks": masks,
        "z_header_scaled": z_header_scaled,
        "header": {
            "nx": nx,
            "ny": ny,
            "bits": bits,
            "min_point": min_point,
            "max_point": max_point,
            "non_measured_points": non_measured,
            "missing_code_applied": missing_code,
            "payload_offset": payload_offset,
            "spacing_xyz": list(spacing),
            "step_units": step_units,
            "units": units,
            "ratios": list(ratios),
            "offsets": list(offsets),
            "header_sha256": sha256_bytes(h),
            "header_interpretation": "audited third-party SUR layout; code step is encoding evidence, not instrument accuracy",
        },
    }


def _read_txt(source: PreparedSource) -> dict[str, np.ndarray | dict[str, Any]]:
    row = source.row
    kwargs = {"dtype": np.float64, "ndmin": 2}
    delimiter = row.get("delimiter", "whitespace")
    if delimiter == "comma":
        kwargs["delimiter"] = ","
    elif delimiter != "whitespace":
        raise ValueError(f"{row['source_id']}: delimiter must be whitespace or comma")
    limit = row.get("max_rows")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError(f"{row['source_id']}: max_rows must be a positive integer")
    try:
        arr = np.loadtxt(source.path, max_rows=limit, **kwargs)
    except Exception as exc:
        raise ValueError(f"{row['source_id']}: invalid numeric TXT: {exc}") from exc
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError(f"{row['source_id']}: TXT must be nonempty and finite")
    schema = row["column_schema"]
    if schema == "xyz_unknown_aux":
        if arr.shape[1] != 4:
            raise ValueError(f"{row['source_id']}: xyz_unknown_aux requires exactly four columns")
        return {
            "values_raw": arr[:, :3],
            "validity_mask": np.isfinite(arr[:, :3]).all(axis=1),
            "sensitivity_masks": {},
            "unknown_aux": arr[:, 3].copy(),
            "header": {
                "column_schema": schema,
                "truncated": limit is not None,
                "rows_read": len(arr),
            },
        }
    if arr.shape[1] != 1:
        raise ValueError(f"{row['source_id']}: profile requires exactly one column")
    sampling = row["xy_sampling"]
    if isinstance(sampling["start"], bool) or isinstance(sampling["step"], bool):
        raise ValueError(f"{row['source_id']}: XY sampling values must be numeric, not bool")
    if not isinstance(sampling["unit"], str) or sampling["unit"] not in _UNITS:
        raise ValueError(f"{row['source_id']}: profile XY sampling unit is invalid")
    if row["units"]["xy"] != sampling["unit"]:
        raise ValueError(f"{row['source_id']}: XY sampling unit conflicts with declared XY unit")
    start, step = float(sampling["start"]), float(sampling["step"])
    if not math.isfinite(start) or not math.isfinite(step) or step <= 0:
        raise ValueError(f"{row['source_id']}: invalid explicit profile sampling")
    x = start + np.arange(len(arr), dtype=np.float64) * step
    return {
        "values_raw": np.column_stack((x, arr[:, 0])),
        "validity_mask": np.isfinite(arr[:, 0]),
        "sensitivity_masks": {},
        "header": {
            "column_schema": schema,
            "xy_sampling": sampling,
            "truncated": limit is not None,
            "rows_read": len(arr),
        },
    }


def _materialize(source: PreparedSource) -> dict[str, Any]:
    row = source.row
    if row["format"] == "royal_csv_zip":
        values = _load_csv_matrix(source.member_bytes or b"", row["source_id"])
        return {
            "values_raw": values,
            "validity_mask": np.isfinite(values),
            "sensitivity_masks": {},
            "header": {
                "member": row["member"],
                "shape": list(values.shape),
                "processing_stage": "unresolved",
            },
        }
    if row["format"] in {"sur", "sur_zip"}:
        return _read_sur(source)
    return _read_txt(source)


def import_study_data(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Preflight every input, then write one NPZ and provenance JSON per source."""
    requested_manifest = Path(manifest_path).resolve()
    manifest_hash = sha256_file(requested_manifest)
    prepared = validate_import_manifest(requested_manifest)
    if sha256_file(prepared.manifest_path) != manifest_hash:
        raise ValueError("import manifest changed during preflight")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    # Parse all sources before creating output, so malformed late sources leave no partial directory.
    loaded = []
    for source in prepared.sources:
        payload = _materialize(source)
        if sha256_file(source.path) != source.source_hash:
            raise ValueError(f"{source.row['source_id']}: source changed during import preflight")
        loaded.append((source, payload))
    out.mkdir(parents=True, exist_ok=False)
    receipts = []
    try:
        if sha256_file(prepared.manifest_path) != manifest_hash:
            raise ValueError("import manifest changed before output write")
        implementation = _implementation_receipt()
        for source, payload in loaded:
            row = source.row
            sid = row["source_id"]
            arrays: dict[str, np.ndarray] = {
                k: np.asarray(v) for k, v in payload.items() if isinstance(v, np.ndarray)
            }
            for mask_name, mask in payload.get("sensitivity_masks", {}).items():
                arrays[f"sensitivity_masks__{mask_name}"] = np.asarray(mask, dtype=bool)
            metadata = {
                "format": FORMAT,
                "manifest_path": str(prepared.manifest_path),
                "manifest_sha256": manifest_hash,
                "implementation": implementation,
                "source_id": sid,
                "source_path": str(source.path),
                "source_sha256": source.source_hash,
                "member": row.get("member"),
                "member_sha256": source.member_hash,
                "exposure": row["exposure"],
                "group_status": row["group_status"],
                "group_id": row.get("group_id"),
                "material_id": row["material_id"],
                "role": row["role"],
                "units": row["units"],
                "unit_evidence": row["unit_evidence"],
                "processing_chain": row["processing_chain"],
                "allowed_metrics": row["allowed_metrics"],
                "header": payload["header"],
                "validity_policy": "finite coordinates plus only format-declared missing code; candidate masks remain separate",
                "sensitivity_mask_keys": sorted(payload.get("sensitivity_masks", {}).keys()),
                "independent_validation_eligible": False,
                "metric_export_eligible": False,
            }
            npz_path = out / f"{sid}.npz"
            json_path = out / f"{sid}.provenance.json"
            np.savez_compressed(npz_path, **arrays)
            metadata["npz"] = npz_path.name
            metadata["npz_path"] = str(npz_path)
            metadata["npz_sha256"] = sha256_file(npz_path)
            json_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            receipts.append(
                {
                    "source_id": sid,
                    "npz": npz_path.name,
                    "npz_sha256": sha256_file(npz_path),
                    "provenance": json_path.name,
                    "provenance_sha256": sha256_file(json_path),
                    "rows": int(np.asarray(payload["values_raw"]).shape[0]),
                }
            )
        receipt = {
            "format": FORMAT,
            "manifest": str(prepared.manifest_path),
            "manifest_sha256": manifest_hash,
            "implementation": implementation,
            "sources": receipts,
            "admission": "import integrity only; no independent holdout or metric export granted",
        }
        if sha256_file(prepared.manifest_path) != manifest_hash:
            raise ValueError("import manifest changed during output write")
        (out / "import-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return receipt
    except Exception:
        # Outputs are exclusively within a new directory created by this call.
        import shutil

        shutil.rmtree(out)
        raise
