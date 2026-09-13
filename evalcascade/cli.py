from __future__ import annotations

import argparse
import json
import sys

from evalcascade.config import load_config
from evalcascade.engine import run_evaluation
from evalcascade.report import render_terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalcascade",
        description="Run Evaluation Cascade against a versioned dataset.",
    )
    parser.add_argument("--dataset", help="Dataset name under datasets/")
    parser.add_argument("--config", default="configs/ci.yaml", help="YAML config path")
    parser.add_argument("--baseline", help="Baseline run id or dataset name")
    parser.add_argument("--adapter", help="mock | ollama | cognitive | extraction | *-mock")
    parser.add_argument("--model", help="Ollama model name")
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit 1 when a critical gate fails")
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--store", help="SQLite path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
    parser.add_argument(
        "--prompt-variant",
        choices=["canonical", "alternate", "both"],
        help="Use the canonical forced-choice phrasing, the alternate phrasing, or both",
    )
    parser.add_argument(
        "--consistency-repeats",
        type=int,
        help="Resample each item this many times and majority-vote (use n>1 with temperature>0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(
        args.config,
        dataset=args.dataset,
        baseline=args.baseline,
        adapter=args.adapter,
        model=args.model,
        fail_on_gate=args.fail_on_gate or None,
        output_json=args.output,
        store_path=args.store,
        prompt_variant=args.prompt_variant,
        consistency_repeats=args.consistency_repeats,
    )
    if args.fail_on_gate:
        config.fail_on_gate = True
    report = run_evaluation(config)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_terminal(report))
        if config.output_json:
            print(f"Wrote {config.output_json}")
    if config.fail_on_gate and report.get("overall_status") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
