"""Score a file of applications, explain High-risk ones, and write the results."""

from pathlib import Path

from .data import load_applications, write_csv
from .explain import generate_explanation
from .scoring import score_application


def run(input_path: Path, output_path: Path) -> dict:
    band_counts = {"Low": 0, "Medium": 0, "High": 0}
    rows = []

    for record in load_applications(input_path):
        scored = score_application(record)
        band_counts[scored.risk_band] += 1

        explanation = ""
        if scored.risk_band == "High":
            explanation = generate_explanation(scored)

        rows.append(
            {
                **record,
                "risk_score": scored.score,
                "risk_band": scored.risk_band,
                "ai_explanation": explanation,
            }
        )

    write_csv(rows, output_path)
    return band_counts
