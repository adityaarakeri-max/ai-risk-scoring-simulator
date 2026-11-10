"""Score a portfolio of applications, explain High-risk ones, and write the results."""

from pathlib import Path

import pandas as pd

from .data import load_applications, write_csv
from .explain import generate_explanation
from .scoring import BANDS, DEFAULT_SCORECARD, Scorecard


def score_portfolio(df: pd.DataFrame, scorecard: Scorecard = DEFAULT_SCORECARD) -> pd.DataFrame:
    scored = [scorecard.score(record) for record in df.to_dict("records")]

    out = df.copy()
    out["risk_score"] = [s.score for s in scored]
    out["risk_band"] = [s.risk_band for s in scored]
    for i, factor in enumerate(scorecard.factors):
        out[f"points_{factor.key}"] = [s.factors[i].points for s in scored]
    out["reason_codes"] = [";".join(r.reason_code for r in s.principal_reasons()) for s in scored]
    return out


def run(input_path: Path, output_path: Path, scorecard: Scorecard = DEFAULT_SCORECARD) -> dict:
    df = load_applications(input_path)
    out = score_portfolio(df, scorecard)

    explanations = []
    for record, band in zip(df.to_dict("records"), out["risk_band"], strict=True):
        explanations.append(generate_explanation(scorecard.score(record)) if band == "High" else "")
    out["ai_explanation"] = explanations

    write_csv(out, output_path)
    counts = out["risk_band"].value_counts()
    return {band: int(counts.get(band, 0)) for band in BANDS}
