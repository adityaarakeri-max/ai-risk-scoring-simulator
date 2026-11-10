"""Command-line interface: `python -m risk_simulator <command>`."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import data, pipeline, validation
from .scoring import DEFAULT_SCORECARD


def _generate(args: argparse.Namespace) -> None:
    df = data.generate_portfolio(n=args.n, seed=args.seed)
    data.write_csv(df, args.out)
    print(
        f"Wrote {len(df)} synthetic applications to {args.out} "
        f"(simulated default rate {df['defaulted'].mean():.1%})"
    )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"No input file at {path}. Create one with `python -m risk_simulator generate`."
        )


def _score(args: argparse.Namespace) -> None:
    _require_file(args.input)
    try:
        counts = pipeline.run(args.input, args.out)
    except data.DataValidationError as exc:
        raise SystemExit(f"{args.input}: {exc}") from exc
    total = sum(counts.values())
    print(f"Scored {total} applications -> {args.out}")
    print("  ".join(f"{band}: {n}" for band, n in counts.items()))


def _validate(args: argparse.Namespace) -> None:
    if args.generate:
        df = data.generate_portfolio(n=args.generate, seed=args.seed)
        source = f"{args.generate:,} freshly simulated applications (seed {args.seed})"
    else:
        _require_file(args.input)
        try:
            df = data.load_applications(args.input)
        except data.DataValidationError as exc:
            raise SystemExit(f"{args.input}: {exc}") from exc
        source = f"{args.input} ({len(df):,} applications)"
    if "defaulted" not in df.columns:
        raise SystemExit("Validation needs a 'defaulted' column (1 = defaulted, 0 = repaid).")

    scored = pipeline.score_portfolio(df)
    overall = validation.discrimination(scored["defaulted"], scored["risk_score"])
    table = validation.band_table(scored)
    bench = validation.benchmark(df, seed=args.seed)

    print(f"Validation on {source}, {df['defaulted'].mean():.1%} defaulted\n")
    print(f"Full sample   Gini {overall.gini:.3f}   AUC {overall.auc:.3f}   KS {overall.ks:.3f}\n")

    print(f"{'Band':<8}{'Apps':>7}{'Share':>8}{'Default rate':>14}")
    for band, row in table.iterrows():
        rate = "-" if row["applications"] == 0 else f"{row['default_rate']:.1%}"
        print(f"{band:<8}{int(row['applications']):>7}{row['share']:>8.1%}{rate:>14}")
    print(f"Rank-ordered: {'yes' if validation.is_rank_ordered(table) else 'NO'}\n")

    print(f"Challenger benchmark on a {bench.test_size:,}-application test set")
    print(f"  Scorecard (current weights)   Gini {bench.scorecard.gini:.3f}")
    print(f"  Scorecard (suggested weights) Gini {bench.reweighted.gini:.3f}")
    print(f"  Logistic regression           Gini {bench.logistic.gini:.3f}\n")

    print(f"{'Factor':<24}{'Current':>9}{'Suggested':>11}")
    for factor in DEFAULT_SCORECARD.factors:
        suggested = bench.suggested_weights[factor.key]
        print(f"{factor.label:<24}{factor.max_points:>9}{suggested:>11}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="risk_simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create a synthetic applications CSV")
    gen.add_argument("--n", type=int, default=1000, help="number of applications")
    gen.add_argument("--seed", type=int, default=42, help="random seed (same seed, same data)")
    gen.add_argument("--out", type=Path, default=Path("data/sample_applications.csv"))
    gen.set_defaults(func=_generate)

    score = sub.add_parser("score", help="score an applications CSV")
    score.add_argument("input", type=Path, nargs="?", default=Path("data/sample_applications.csv"))
    score.add_argument("--out", type=Path, default=Path("output/risk_scored_results.csv"))
    score.set_defaults(func=_score)

    val = sub.add_parser("validate", help="measure scorecard performance against outcomes")
    val.add_argument("input", type=Path, nargs="?", default=Path("data/sample_applications.csv"))
    val.add_argument("--generate", type=int, metavar="N", help="validate on N fresh simulated apps")
    val.add_argument("--seed", type=int, default=42)
    val.set_defaults(func=_validate)

    return parser


def main(argv: list | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.func(args)
