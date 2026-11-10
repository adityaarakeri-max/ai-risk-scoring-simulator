"""Command-line interface: `python -m risk_simulator <command>`."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import data, pipeline


def _generate(args: argparse.Namespace) -> None:
    rows = data.generate_records(n=args.n, seed=args.seed)
    data.write_csv(rows, args.out)
    print(f"Wrote {len(rows)} synthetic applications to {args.out}")


def _score(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise SystemExit(
            f"No input file at {args.input}. Create one with `python -m risk_simulator generate`."
        )
    counts = pipeline.run(args.input, args.out)
    total = sum(counts.values())
    print(f"Scored {total} applications -> {args.out}")
    print("  ".join(f"{band}: {n}" for band, n in counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="risk_simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create a synthetic applications CSV")
    gen.add_argument("--n", type=int, default=200, help="number of applications")
    gen.add_argument("--seed", type=int, default=42, help="random seed (same seed, same data)")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic_loans.csv"))
    gen.set_defaults(func=_generate)

    score = sub.add_parser("score", help="score an applications CSV")
    score.add_argument("input", type=Path, nargs="?", default=Path("data/synthetic_loans.csv"))
    score.add_argument("--out", type=Path, default=Path("output/risk_scored_results.csv"))
    score.set_defaults(func=_score)

    return parser


def main(argv: list | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.func(args)
