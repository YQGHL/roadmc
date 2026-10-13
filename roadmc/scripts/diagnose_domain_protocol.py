"""Calibrate or apply a fixed road-domain evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.domain_protocol import (
    DEFAULT_PROTOCOL,
    evaluate_manifest,
    fit_protocol,
    inspect_synthetic_candidates,
    load_manifest,
    load_protocol,
    save_protocol,
    split_real_roads,
)


def _write_json(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="make a deterministic road-level 60/20/20 split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=int(DEFAULT_PROTOCOL["seed"]))

    calibrate = commands.add_parser("calibrate", help="fit normalization and MMD bandwidth from real calibration samples only")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--output-protocol", required=True)
    calibrate.add_argument("--settings", default="", help="JSON object overriding the fixed protocol defaults")

    evaluate = commands.add_parser("evaluate", help="apply a saved protocol without refitting it")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--output", required=True)

    inspect = commands.add_parser("inspect", help="inspect synthetic candidate support without fitting or acceptance")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--settings", default="", help="JSON object overriding protocol defaults for diagnostic geometry")
    inspect.add_argument("--output", required=True)

    args = parser.parse_args()
    samples = load_manifest(args.manifest)
    if args.command == "split":
        result = split_real_roads(samples, seed=args.seed)
        _write_json(result, args.output)
        print(json.dumps({"status": result["status"], "road_count": result.get("road_count", len(result.get("road_assignment", {}))), "output": str(Path(args.output).resolve())}, ensure_ascii=False))
        return
    if args.command == "calibrate":
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8")) if args.settings else None
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("--settings must contain a JSON object")
        protocol = fit_protocol(samples, protocol_options=settings)
        save_protocol(protocol, args.output_protocol)
        print(json.dumps({"status": protocol["status"], "protocol_hash": protocol.get("protocol_hash"), "output": str(Path(args.output_protocol).resolve())}, ensure_ascii=False))
        return
    if args.command == "inspect":
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8")) if args.settings else None
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("--settings must contain a JSON object")
        report = inspect_synthetic_candidates(samples, protocol_options=settings)
        _write_json(report, args.output)
        print(json.dumps({"status": report["status"], "candidate_count": report["synthetic_candidate_count"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))
        return
    protocol = load_protocol(args.protocol)
    report = evaluate_manifest(samples, protocol)
    _write_json(report, args.output)
    print(json.dumps({"status": report["status"], "protocol_hash": report.get("protocol_hash"), "output": str(Path(args.output).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
