"""Import explicitly selected, evidence-bound study sources into native bundles."""

from __future__ import annotations

import argparse
import json

from roadmc.data.real.study_import import import_study_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="must be a new, nonexistent directory")
    args = parser.parse_args()
    result = import_study_data(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
