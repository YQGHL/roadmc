"""Run a frozen procedural realism study, with stage-specific provenance/resume.

Each actual experiment runs in an owned subprocess with a finite timeout. Failed
attempts stay on disk. Resume verifies inputs, executing code and saved outputs;
seen-file comparisons never become fresh independent-road validation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from roadmc.data.synthetic.study_protocol import (
    canonical_hash,
    file_hash,
    read_json,
    validate_manifest,
    validate_protocol,
    validate_stage_assets,
    write_json,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STAGES = (
    "admission",
    "texture_fit",
    "texture_benchmark",
    "geometry_material",
    "pothole_native",
)
# Frozen-result diagnostics are opt-in, so old commands keep their workload.
STAGES = DEFAULT_STAGES + ("pothole_outline_fairness",)
DEPENDENCIES = {"texture_benchmark": ("texture_fit",)}
COMMON = [
    "roadmc/scripts/run_realism_study.py",
    "roadmc/data/synthetic/study_protocol.py",
    "pyproject.toml",
    "uv.lock",
]
TEXTURE = [
    "roadmc/data/synthetic/joint_texture_calibration.py",
    "roadmc/data/synthetic/procedural_texture.py",
    "roadmc/data/synthetic/spectral_synthesis.py",
    "roadmc/data/synthetic/texture_shape_calibration.py",
    "roadmc/data/synthetic/am_envelope.py",
    "roadmc/scripts/calibrate_texture_shape.py",
    "roadmc/scripts/benchmark_texture_envelopes.py",
]
GEOMETRY = [
    "roadmc/data/synthetic/labels.py",
    "roadmc/scripts/realism_experiments.py",
    "roadmc/data/synthetic/surface_replay.py",
    "roadmc/data/synthetic/pothole_contour.py",
    "roadmc/data/synthetic/pothole_profile.py",
    "roadmc/data/synthetic/material_diagnostics.py",
    "roadmc/data/synthetic/material_partition.py",
    "roadmc/data/synthetic/outline_diagnostics.py",
    "roadmc/data/synthetic/surface_tile.py",
    "roadmc/data/synthetic/crack_graph.py",
    "roadmc/data/synthetic/line_profiler.py",
    "roadmc/data/synthetic/config.py",
    "roadmc/data/synthetic/primitives.py",
]
SOURCES = {
    "admission": COMMON,
    "texture_fit": COMMON + TEXTURE,
    "texture_benchmark": COMMON + TEXTURE,
    "geometry_material": COMMON + GEOMETRY + TEXTURE[1:5],
    "pothole_native": COMMON + GEOMETRY + ["roadmc/scripts/diagnose_pothole_profiles.py"],
    "pothole_outline_fairness": COMMON + GEOMETRY + ["roadmc/scripts/diagnose_pothole_profiles.py"],
}


def source_hashes(stage):
    return {name: file_hash(REPO / name) for name in SOURCES[stage]}


def execution_environment(stage):
    """Describe this interpreter and loaded versions used by the stage worker."""
    # Admission uses only the standard library. Numerical stages also import
    # plotting/image readers, so capture their versions without importing torch.
    libraries = () if stage == "admission" else ("numpy", "scipy", "matplotlib", "PIL")
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "libraries": {name: importlib.import_module(name).__version__ for name in libraries},
    }


def stage_fingerprint(input_hash, sources, environment, dependencies):
    return canonical_hash(
        {
            "inputs": input_hash,
            "sources": sources,
            "execution_environment": environment,
            "dependencies": dependencies,
        }
    )


def artifact_hashes(directory):
    return {
        str(path.relative_to(directory)).replace("\\", "/"): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verified_stage(state, stage, output, input_hash):
    """Verify dependencies too, including ones omitted from this invocation."""
    row = state["stages"].get(stage, {})
    if row.get("status") != "completed":
        return False
    dependencies = DEPENDENCIES.get(stage, ())
    if not all(verified_stage(state, dep, output, input_hash) for dep in dependencies):
        return False
    try:
        environment = execution_environment(stage)
        if row.get("execution_environment") != environment:
            return False  # Legacy rows without environment evidence are stale.
        fingerprint = stage_fingerprint(
            input_hash,
            source_hashes(stage),
            environment,
            {dep: state["stages"][dep]["result_sha256"] for dep in dependencies},
        )
        directory = output / row["artifact_dir"]
        return (
            row.get("fingerprint") == fingerprint
            and directory.is_dir()
            and artifact_hashes(directory) == row.get("artifact_hashes")
            and file_hash(output / row["result_path"]) == row.get("result_sha256")
        )
    except (KeyError, FileNotFoundError, ImportError):
        return False


def _worker(stage, manifest_path, protocol_path, directory, fit_result=None):
    manifest = read_json(manifest_path)
    protocol = validate_protocol(read_json(protocol_path))
    admission = validate_manifest(manifest, manifest_path.parent)
    validate_stage_assets(stage, admission)
    assets = {key: Path(value) for key, value in admission["resolved_assets"].items()}
    texture = protocol["texture"]
    if stage == "admission":
        directory.mkdir(parents=True)
        result = admission
    elif stage == "texture_fit":
        from roadmc.data.synthetic.joint_texture_calibration import fit_joint_texture

        result = fit_joint_texture(
            assets["texture_archive"],
            assets["texture_split"],
            assets["texture_prior"],
            directory,
            seeds=tuple(texture["seeds"]),
            multipliers=tuple(texture["multipliers"]),
            iterations=texture["iterations"],
            tolerance=texture["tolerance"],
        )
    elif stage == "texture_benchmark":
        from roadmc.data.synthetic.joint_texture_calibration import benchmark_joint_texture

        if fit_result is None:
            raise ValueError("texture benchmark requires completed frozen fit result")
        result = benchmark_joint_texture(
            assets["texture_archive"],
            assets["texture_split"],
            read_json(fit_result),
            directory,
            seeds=tuple(texture["seeds"]),
        )
    elif stage == "geometry_material":
        from roadmc.scripts.realism_experiments import run_geometry_material_study

        result = run_geometry_material_study(directory, protocol)
    elif stage == "pothole_native":
        from roadmc.scripts.realism_experiments import run_pothole_native_study

        result = run_pothole_native_study(
            assets["pothole_frame_01"].parent.parent, directory, protocol
        )
    elif stage == "pothole_outline_fairness":
        from roadmc.data.synthetic.outline_diagnostics import run_pothole_outline_fairness_study

        result = run_pothole_outline_fairness_study(
            assets["pothole_frame_01"].parent.parent,
            directory,
            assets["pothole_frozen_metrics"],
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    write_json(directory / "execution_environment.json", execution_environment(stage))
    write_json(directory / "stage_result.json", result)


def _summary(output, state):
    lines = [
        "# RoadMC 受控真实性研究",
        "",
        "状态按阶段记录。软件执行成功、形态对照与独立道路真实性验收分别判断。",
        "当前真实域状态：not_calibrated / not_evaluated。",
        "",
        "|阶段|状态|结果|",
        "|---|---|---|",
    ]
    for stage, row in state["stages"].items():
        relative = row.get("result_path")
        target = f"[JSON]({relative})" if relative else row.get("error", row.get("reason", "—"))
        lines.append(f"|{stage}|{row['status']}|{target}|")
    fit = state["stages"].get("texture_fit", {})
    if fit.get("status") == "completed":
        result = read_json(output / fit["result_path"])
        lines += [
            "",
            "## 训练内部候选决定",
            "",
            "交叉验证只使用原训练文件；历史谱参数也来自该数据体系，因此不称为独立道路验证。",
        ]
        for category, decision in result.get("decisions", {}).items():
            lines.append(
                f"- {category}: `{decision.get('selected_candidate_id', 'unknown')}`；详见冻结配置和逐折指标。"
            )
    benchmark = state["stages"].get("texture_benchmark", {})
    if benchmark.get("status") == "completed":
        result = read_json(output / benchmark["result_path"])
        lines += [
            "",
            "## 已见文件基准",
            "",
            "以下不是新留出验收；不据结果继续调参。",
            "",
            "```json",
            json.dumps(result.get("equal_class_summary", {}), ensure_ascii=False, indent=2),
            "```",
        ]
    lines += [
        "",
        "## 执行边界",
        "",
        "原始数据不修改，真实样本不复制进程序化表面。默认生产参数不自动替换。",
        "单位未知、病害身份未核实、仅有候选掩膜或已见文件时，保留相应限制。",
        "数据核查和外部访问限制见研究根目录data_audit；实际源码、输入和产物哈希见run_state.json。",
    ]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    manifest_path, protocol_path, output, *, stages=DEFAULT_STAGES, resume=False, timeout_s=1800
):
    manifest_path, protocol_path, output = (
        Path(path).resolve() for path in (manifest_path, protocol_path, output)
    )
    manifest, protocol = read_json(manifest_path), validate_protocol(read_json(protocol_path))
    admission = validate_manifest(manifest, manifest_path.parent)
    if (
        not stages
        or len(set(stages)) != len(stages)
        or any(stage not in STAGES for stage in stages)
    ):
        raise ValueError("stages must be unique known stage names")
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or timeout_s <= 0:
        raise ValueError("positive integer stage timeout required")
    selected = [stage for stage in STAGES if stage in stages]
    for stage in selected:
        validate_stage_assets(stage, admission)
    input_hash = canonical_hash({"manifest": manifest, "protocol": protocol})
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"output exists; use --resume with matching frozen inputs: {output}"
            )
        state = read_json(output / "run_state.json")
        if state["input_hash"] != input_hash:
            raise ValueError(
                "resume input/protocol differs from frozen study; use a new output directory"
            )
    else:
        if resume:
            raise FileNotFoundError("cannot resume a nonexistent study")
        output.mkdir(parents=True)
        write_json(output / "frozen_manifest.json", manifest)
        write_json(output / "frozen_protocol.json", protocol)
        state = {
            "schema_version": 1,
            "input_hash": input_hash,
            "stages": {},
            "history": [],
            "calibration_status": "not_calibrated",
            "real_domain_status": "not_evaluated",
        }
    (output / "logs").mkdir(exist_ok=True)
    # Completion on disk is not enough when its code or dependencies changed.
    for stage in STAGES:
        row = state["stages"].get(stage, {})
        if row.get("status") == "completed" and not verified_stage(
            state, stage, output, input_hash
        ):
            row.update(
                status="stale",
                reason="input, source, execution environment, dependency or artifact hashes changed",
            )
    for stage in selected:
        old = state["stages"].get(stage, {})
        dependencies = DEPENDENCIES.get(stage, ())
        if not all(verified_stage(state, dep, output, input_hash) for dep in dependencies):
            state["stages"][stage] = {
                "status": "blocked_dependency",
                "reason": f"requires current, hash-verified {dependencies}",
            }
            write_json(output / "run_state.json", state)
            continue
        sources = source_hashes(stage)
        environment_info = execution_environment(stage)
        dependency_hashes = {dep: state["stages"][dep]["result_sha256"] for dep in dependencies}
        fingerprint = stage_fingerprint(
            input_hash, sources, environment_info, dependency_hashes
        )
        if verified_stage(state, stage, output, input_hash):
            old["cache_verified"] = True
            write_json(output / "run_state.json", state)
            print(f"REUSE {stage}: sources, inputs, environment and artifact hashes verified", flush=True)
            continue
        attempt = 1 + sum(item["stage"] == stage for item in state["history"])
        relative = Path("artifacts") / stage / f"attempt-{attempt:02d}"
        directory = output / relative
        snapshot = output / "executing_source" / stage / f"attempt-{attempt:02d}"
        for name in sources:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        log = output / "logs" / f"{stage}-{attempt:02d}.log"
        row = {
            "status": "running",
            "attempt": attempt,
            "fingerprint": fingerprint,
            "source_hashes": sources,
            "execution_environment": environment_info,
            "artifact_dir": relative.as_posix(),
            "log": log.relative_to(output).as_posix(),
        }
        state["stages"][stage] = row
        write_json(output / "run_state.json", state)
        command = [
            sys.executable,
            "-m",
            "roadmc.scripts.run_realism_study",
            "--manifest",
            str(manifest_path),
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(directory),
            "--worker-stage",
            stage,
        ]
        if "texture_fit" in dependencies:
            command += ["--fit-result", str(output / state["stages"]["texture_fit"]["result_path"])]
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", ROADMC_GENERATOR_NO_TORCH="1")
        started = time.perf_counter()
        print(f"START {stage} attempt {attempt}", flush=True)
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    cwd=REPO,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
            row["exit_code"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f"stage exited {process.returncode}; see {log}")
            if source_hashes(stage) != sources:
                raise RuntimeError("executing source changed during stage; result is not accepted")
            if (
                execution_environment(stage) != environment_info
                or read_json(directory / "execution_environment.json") != environment_info
            ):
                raise RuntimeError("execution environment changed during stage; result is not accepted")
            result_path = directory / "stage_result.json"
            read_json(result_path)
            row.update(
                {
                    "status": "completed",
                    "result_path": result_path.relative_to(output).as_posix(),
                    "result_sha256": file_hash(result_path),
                    "artifact_hashes": artifact_hashes(directory),
                    "cache_verified": False,
                }
            )
        except (RuntimeError, subprocess.TimeoutExpired, ValueError, FileNotFoundError) as error:
            row.update({"status": "failed", "error": str(error)})
        row["elapsed_seconds"] = time.perf_counter() - started
        state["history"].append(
            {
                "stage": stage,
                "attempt": attempt,
                "status": row["status"],
                "fingerprint": fingerprint,
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
        write_json(output / "run_state.json", state)
        print(f"{row['status'].upper()} {stage}: {row['elapsed_seconds']:.1f}s", flush=True)
    state["requested_stages"] = selected
    state["all_requested_completed"] = all(
        state["stages"][stage]["status"] == "completed" for stage in selected
    )
    state["admission"] = admission
    write_json(output / "run_state.json", state)
    _summary(output, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument("--stage-timeout-s", type=int, default=1800)
    parser.add_argument("--worker-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--fit-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_stage:
        _worker(
            args.worker_stage,
            args.manifest.resolve(),
            args.protocol.resolve(),
            args.output_dir.resolve(),
            args.fit_result,
        )
    else:
        result = run_study(
            args.manifest,
            args.protocol,
            args.output_dir,
            stages=tuple(args.stages),
            resume=args.resume,
            timeout_s=args.stage_timeout_s,
        )
        if not result["all_requested_completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
