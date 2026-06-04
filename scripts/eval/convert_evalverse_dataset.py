#!/usr/bin/env python3
"""Convert EvalVerse raw JSON/JSONL into ARR canonical pairwise JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_pipeline.datasets.evalverse_adapter import load_evalverse_dataset, save_arr_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EvalVerse dataset to ARR JSON.")
    parser.add_argument("--input", required=True, help="EvalVerse raw JSON or JSONL path")
    parser.add_argument("--output", required=True, help="Output ARR JSON path")
    args = parser.parse_args()

    samples = load_evalverse_dataset(args.input)
    out_path = save_arr_dataset(samples, args.output)
    print(f"Wrote {len(samples)} ARR samples to {out_path}")


if __name__ == "__main__":
    main()
