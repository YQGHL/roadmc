"""Freeze explicit metric crack graph specifications into a schema-v2 plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roadmc.data.synthetic.crack_graph import (
    build_alligator_graph,
    build_bezier_graph,
    build_block_graph,
)
from roadmc.data.synthetic.surface_replay import validate_plan


def _build(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec.pop("builder", None)
    if kind == "bezier":
        return build_bezier_graph(**spec)
    if kind == "alligator":
        return build_alligator_graph(**spec)
    if kind == "block":
        return build_block_graph(**spec)
    raise ValueError(f"unknown graph builder: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="explicit builder request JSON")
    parser.add_argument("--output", type=Path, required=True, help="frozen plan JSON destination")
    args = parser.parse_args()
    request = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or request.get("schema_version") != 2:
        parser.error("spec root must be a schema_version=2 plan")
    plan = dict(request)
    frozen_events = []
    for event in request.get("events", []):
        required_event = {
            "instance_id",
            "type",
            "seed",
            "label",
            "builder_spec",
            "profile",
            "texture",
            "variant",
        }
        if not isinstance(event, dict) or set(event) not in (
            required_event,
            required_event | {"severity"},
        ):
            parser.error(
                "builder events require instance_id/type/seed/label/builder_spec/profile/texture/variant and optional severity"
            )
        if event["type"] != "crack":
            parser.error("this generator currently builds crack graph events only")
        builder_spec = dict(event["builder_spec"])
        builder_name = builder_spec.get("builder")
        allowed_variants = {
            "bezier": {"longitudinal", "transverse"},
            "alligator": {"alligator"},
            "block": {"block"},
        }
        if event["variant"] not in allowed_variants.get(builder_name, set()):
            parser.error(
                f"builder {builder_name!r} cannot produce declared variant {event['variant']!r}"
            )
        if "seed" in builder_spec and builder_spec["seed"] != event["seed"]:
            parser.error("builder seed, when present, must equal the explicit event seed")
        builder_spec["seed"] = event["seed"]
        graph = _build(builder_spec)
        params = {
            "variant": event["variant"],
            "graph": graph,
            "profile": event["profile"],
            "texture": event["texture"],
        }
        if event.get("severity") is not None:
            params["severity"] = event["severity"]
        frozen_events.append(
            {
                "instance_id": event["instance_id"],
                "type": "crack",
                "seed": event["seed"],
                "label": event["label"],
                "parameters": params,
            }
        )
    plan["events"] = frozen_events
    plan_hash, _ = validate_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "plan_hash": plan_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
