"""Frozen KIT native-grid preparation and spatially separated repeat diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadmc.data.real.kit_concrete import (
    align_repeat_grids,
    common_support_repeat_scores,
    native_height_grid,
    read_kit_mesh,
    resample_native_grid,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.audit / "paper_followup"
    groups = json.loads((evidence / "kit_source_groups.json").read_text(encoding="utf-8"))["files"]
    old = json.loads((args.audit / "pilot_results.json").read_text(encoding="utf-8"))
    exposed = {r["member"]: r["sha256"] for r in old["kit_ply"]}
    members = groups if args.all_meshes else [r for r in groups if r["member"] in exposed]
    protocol = {
        "schema": "roadmc.kit-native-grid.v1",
        "tar": str(args.tar.resolve()),
        "source_size": args.tar.stat().st_size,
        "source_mtime_ns": args.tar.stat().st_mtime_ns,
        "source_previous_hash_receipt": str(args.audit / "source_hashes.json"),
        "previous_hash_receipt_sha256": sha(args.audit / "source_hashes.json"),
        "members": [dict(r, expected_sha256=exposed.get(r["member"])) for r in members],
        "grid": {"spacing": 0.5, "x_bounds": [-90, 90], "y_bounds": [-130, 130]},
        "repeat": {
            "fit_x": [-90, -10],
            "evaluation_x": [10, 90],
            "angle_bound": 6.0,
            "translation_bound": 15.0,
        },
        "units": "native source units; mm is inference from paper geometry, not a declaration; metric calibration not established",
        "statistical_scope": "19 files map to16 patches,4 locations,2 roads; repeated meshes never added as independent patches",
        "missingness": "Mean finite point residual per bin; store original counts; no filled heights in statistical summaries",
        "model_fitting": "Rigid registration uses fit strip only; no synthetic model selected here",
        "code": [
            {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
            {
                "path": str(Path(__file__).resolve().parents[1] / "data/real/kit_concrete.py"),
                "sha256": sha(Path(__file__).resolve().parents[1] / "data/real/kit_concrete.py"),
            },
        ],
        "evidence": [
            {"path": str(evidence / n), "sha256": sha(evidence / n)}
            for n in ["kit_source_groups.json", "paper_evidence.json"]
        ],
    }
    write(args.output / "protocol.json", protocol)


def run(args):
    path = args.output / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (args.output / "results.json").exists():
        raise FileExistsError("completed results exist; use a new output directory")
    for r in protocol["code"] + protocol["evidence"]:
        if sha(r["path"]) != r["sha256"]:
            raise ValueError(f"frozen code/evidence changed: {r['path']}")
    tar = Path(protocol["tar"])
    if (tar.stat().st_size, tar.stat().st_mtime_ns) != (
        protocol["source_size"],
        protocol["source_mtime_ns"],
    ):
        raise ValueError("source changed since freezing")
    start = time.monotonic()
    arrays, rows = {}, []
    for member in protocol["members"]:
        name = PurePosixPath(member["member"]).stem
        pts, _, source = read_kit_mesh(tar, member["member"], member["expected_sha256"])
        grid = native_height_grid(pts, **protocol["grid"])
        target = args.output / (name + "_grid.npz")
        np.savez_compressed(target, **grid)
        g, count = grid["grid_height_source_units"], grid["grid_count"]
        finite = g[np.isfinite(g)]
        if not len(finite):
            raise ValueError(f"empty fixed ROI: {name}")
        row = dict(
            member,
            source=source,
            grid_path=str(target),
            grid_sha256=sha(target),
            source_bounds_source_units=[pts.min(axis=0).tolist(), pts.max(axis=0).tolist()],
            grid_shape=list(g.shape),
            support_fraction=float((count > 0).mean()),
            count_quantiles=np.quantile(count[count > 0], [0, 0.25, 0.5, 0.75, 1]).tolist(),
            residual_rms_source_units=float(np.sqrt(np.mean(finite**2))),
            conditional_rms_m_if_mm=float(np.sqrt(np.mean(finite**2))) * 0.001,
            plane_coefficients=grid["plane_coefficients"].tolist(),
        )
        arrays[name] = g
        rows.append(row)
        # Do not print L2 numerical descriptors before any downstream fit protocol is frozen.
        print(
            json.dumps(
                {"member_prepared": name, "grid_shape": list(g.shape), "crc_verified": True}
            ),
            flush=True,
        )
    repeats = []
    for original in ("A5_L1_M2", "A6_L1_M2"):
        if original not in arrays or original + "_rep" not in arrays:
            continue
        a, b = arrays[original], arrays[original + "_rep"]
        result, aligned = align_repeat_grids(a, b, **protocol["repeat"])
        result["pair"] = original
        repeats.append(result)
        np.savez_compressed(
            args.output / (original + "_registration.npz"), reference=a, moving=b, aligned=aligned
        )
        limit = float(np.nanquantile(np.abs(a), 0.98))
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
        for ax, z, title in zip(
            axes,
            (a, b, aligned - a),
            ("Reference", "Repeat before alignment", "Aligned minus reference"),
            strict=True,
        ):
            image = ax.imshow(
                z.T,
                origin="lower",
                extent=[-90, 90, -130, 130],
                vmin=-limit,
                vmax=limit,
                cmap="RdBu_r",
            )
            ax.axvline(-10, color="k", linestyle="--")
            ax.axvline(10, color="k", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("X source units")
        axes[0].set_ylabel("Y source units")
        fig.colorbar(
            image, ax=axes, label="Residual height, source units (mm scale is inferred)", shrink=0.7
        )
        fig.savefig(args.output / (original + "_repeat.png"), dpi=160)
        plt.close(fig)
        print(
            json.dumps(
                {
                    "registration_pair": original,
                    "fit": result["fit_after"],
                    "evaluation": result["evaluation_after"],
                    "common_support": result["common_support"],
                }
            ),
            flush=True,
        )
    results = {
        "schema": "roadmc.kit-native-grid-results.v1",
        "protocol_sha256": sha(path),
        "grids": rows,
        "repeats": repeats,
        "elapsed_s": time.monotonic() - start,
        "scientific_status": "conditional units; same-patch repeat diagnostic; no independent-road validation",
        "no_raw_source_changed": (tar.stat().st_size, tar.stat().st_mtime_ns)
        == (protocol["source_size"], protocol["source_mtime_ns"]),
    }
    write(args.output / "results.json", results)
    write(
        args.output / "grids_manifest.json",
        {
            "protocol_sha256": sha(path),
            "grids": [
                {
                    k: r[k]
                    for k in (
                        "member",
                        "patch_id",
                        "location_id",
                        "surface_finish",
                        "is_repeat",
                        "grid_path",
                        "grid_sha256",
                    )
                }
                for r in rows
            ],
        },
    )


def rescore_saved_repeats(source: Path, output: Path):
    """Add common-support evidence from saved arrays; never reread or refit meshes."""
    source, output = source.resolve(), output.resolve()
    source_result_path, source_protocol_path = source / "results.json", source / "protocol.json"
    previous = json.loads(source_result_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if sha(source_protocol_path) != previous["protocol_sha256"]:
        raise ValueError("saved result does not match its source protocol")
    registration = source_protocol.get("repeat", source_protocol.get("registration"))
    if registration is None:
        raise ValueError("source protocol has no repeat registration regions")
    spacing = source_protocol["grid"]["spacing"]
    regions = {key: registration[key] for key in ("fit_x", "evaluation_x")}
    inputs = [source_result_path, source_protocol_path]
    for row in previous["repeats"]:
        pair = row["pair"]
        if PurePosixPath(pair).name != pair or "\\" in pair or pair in (".", ".."):
            raise ValueError("unsafe saved repeat pair name")
        inputs.append(source / (pair + "_registration.npz"))
    output.mkdir(parents=True, exist_ok=False)
    module = Path(__file__).resolve().parents[1] / "data/real/kit_concrete.py"
    historical_bindings = []
    for entry in source_protocol["code"]:
        path = Path(entry["path"])
        current = sha(path) if path.is_file() else None
        historical_bindings.append(
            dict(
                entry, current_sha256=current, current_matches_historical=current == entry["sha256"]
            )
        )
    protocol = {
        "schema": "roadmc.kit-saved-common-support.v1",
        "operation": "score saved arrays at frozen transform parameters; no registration optimization",
        "raw_source_read": False,
        "source_directory": str(source),
        "spacing_source_units": spacing,
        "regions": regions,
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
        "code": [{"path": str(p), "sha256": sha(p)} for p in (Path(__file__).resolve(), module)],
        "historical_code_binding": historical_bindings,
        "support": "intersection of finite reference/before/after for each FIT/EVAL region",
        "offsets": "both height offsets estimated only on common FIT support",
        "scope": "reference-grid strips after full-patch bbox/plane preprocessing; not a strict dual-domain or raw end-to-end holdout",
        "units": "native source units; mm remains a conditional inference",
    }
    write(output / "protocol.json", protocol)
    rows = []
    for row in previous["repeats"]:
        path = source / (row["pair"] + "_registration.npz")
        with np.load(path, allow_pickle=False) as arrays:
            a, b, saved = arrays["reference"], arrays["moving"], arrays["aligned"]
        parameters = row["parameters_angle_deg_translation_source_units"]
        before = resample_native_grid(b, [0, 0, 0], spacing=spacing)
        after = resample_native_grid(b, parameters, spacing=spacing)
        replay = after + row["height_offset_from_fit_source_units"]
        if not np.array_equal(replay, saved, equal_nan=True):
            raise ValueError(
                f"saved aligned array does not replay at frozen parameters: {row['pair']}"
            )
        rows.append(
            {
                "pair": row["pair"],
                "source_registration_sha256": sha(path),
                "frozen_parameters": parameters,
                "saved_aligned_exactly_replayed": True,
                "original_own_support": {
                    key: row[key]
                    for key in (
                        "fit_before",
                        "fit_after",
                        "evaluation_before",
                        "evaluation_after",
                        "height_offset_from_fit_source_units",
                    )
                },
                "common_support": common_support_repeat_scores(
                    a, before, after, spacing=spacing, **regions
                ),
            }
        )
    write(
        output / "results.json",
        {
            "schema": "roadmc.kit-saved-common-support-results.v1",
            "protocol_sha256": sha(output / "protocol.json"),
            "repeats": rows,
            "registration_reoptimized": False,
            "scientific_scope": protocol["scope"],
        },
    )
    write(
        output / "completion_receipt.json",
        {
            "exit_code": 0,
            "source_inputs_unchanged": all(
                sha(e["path"]) == e["sha256"] for e in protocol["inputs"]
            ),
            "artifacts": [
                {"path": str(output / name), "sha256": sha(output / name)}
                for name in ("protocol.json", "results.json")
            ],
        },
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-meshes", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rescore-common-support", type=Path, metavar="SAVED_STUDY_DIRECTORY")
    args = parser.parse_args()
    if args.freeze:
        if args.tar is None or args.audit is None:
            parser.error("--freeze requires --tar and --audit")
        freeze(args)
    elif args.run:
        run(args)
    else:
        rows = rescore_saved_repeats(args.rescore_common_support, args.output)
        print(
            json.dumps(
                {"exit_code": 0, "saved_pairs_rescored": len(rows), "output": str(args.output)}
            )
        )


if __name__ == "__main__":
    main()
