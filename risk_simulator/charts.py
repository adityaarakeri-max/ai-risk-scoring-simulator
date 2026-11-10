"""Altair chart builders for the Streamlit app.

Colours come from a validated palette. Risk bands use the reserved status
colours and always appear with a text label, so colour never carries meaning
alone. Two-series comparisons use categorical slots 1 and 2, which pass
colour-vision-deficiency separation checks in both light and dark mode.
"""

import altair as alt
import pandas as pd

from .scoring import BANDS, ScoredApplication
from .validation import roc_points

STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

PALETTES = {
    "light": {"series": ("#2a78d6", "#eb6834"), "muted": "#898781", "track": "#e1e0d9"},
    "dark": {"series": ("#3987e5", "#d95926"), "muted": "#898781", "track": "#383835"},
}


def _palette(mode: str) -> dict:
    return PALETTES["dark" if mode == "dark" else "light"]


def _band_color() -> alt.Color:
    return alt.Color(
        "risk_band:N",
        scale=alt.Scale(domain=list(BANDS), range=[STATUS[b] for b in BANDS]),
        legend=alt.Legend(title="Risk band", orient="top"),
    )


def score_meter(score: int, low_max: int, medium_max: int) -> alt.LayerChart:
    zones = pd.DataFrame(
        {
            "risk_band": list(BANDS),
            "start": [0, low_max + 1, medium_max + 1],
            "end": [low_max + 1, medium_max + 1, 101],
        }
    )
    zones["label"] = [
        f"Low 0–{low_max}",
        f"Medium {low_max + 1}–{medium_max}",
        f"High {medium_max + 1}–100",
    ]
    x = alt.X("start:Q", scale=alt.Scale(domain=[0, 101]), axis=None)

    bands = (
        alt.Chart(zones)
        .mark_bar(height=14, opacity=0.85)
        .encode(
            x=x,
            x2="end:Q",
            color=_band_color().legend(None),
            tooltip=[alt.Tooltip("label:N", title="Band")],
        )
    )
    labels = (
        alt.Chart(zones)
        .mark_text(dy=22, fontSize=12)
        .encode(x=alt.X("mid:Q", scale=alt.Scale(domain=[0, 101]), axis=None), text="label:N")
        .transform_calculate(mid="(datum.start + datum.end) / 2")
    )
    marker_data = pd.DataFrame({"score": [score + 0.5], "text": [f"{score}"]})
    marker = (
        alt.Chart(marker_data)
        .mark_tick(thickness=3, size=30)
        .encode(x=alt.X("score:Q", scale=alt.Scale(domain=[0, 101]), axis=None))
    )
    marker_label = (
        alt.Chart(marker_data)
        .mark_text(dy=-22, fontSize=13, fontWeight="bold")
        .encode(x=alt.X("score:Q", scale=alt.Scale(domain=[0, 101]), axis=None), text="text:N")
    )
    return alt.layer(bands, labels, marker, marker_label).properties(height=80)


def factor_contributions(scored: ScoredApplication, mode: str = "light") -> alt.LayerChart:
    colors = _palette(mode)
    data = pd.DataFrame(
        {
            "factor": [f.label for f in scored.factors],
            "value": [f.display_value for f in scored.factors],
            "points": [f.points for f in scored.factors],
            "max_points": [f.max_points for f in scored.factors],
            "code": [f.reason_code for f in scored.factors],
        }
    )
    data["label"] = data["points"].astype(str) + " / " + data["max_points"].astype(str)
    y = alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=200))
    tooltip = [
        alt.Tooltip("factor:N", title="Factor"),
        alt.Tooltip("value:N", title="Value"),
        alt.Tooltip("label:N", title="Points"),
        alt.Tooltip("code:N", title="Reason code"),
    ]

    track = (
        alt.Chart(data)
        .mark_bar(color=colors["track"], cornerRadiusEnd=4, height=16)
        .encode(x=alt.X("max_points:Q", title="Risk points"), y=y, tooltip=tooltip)
    )
    filled = (
        alt.Chart(data)
        .mark_bar(color=colors["series"][0], cornerRadiusEnd=4, height=16)
        .encode(x="points:Q", y=y, tooltip=tooltip)
    )
    text = (
        alt.Chart(data)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(x="max_points:Q", y=y, text="label:N")
    )
    return alt.layer(track, filled, text).properties(height=200)


def score_distribution(scored: pd.DataFrame, low_max: int, medium_max: int) -> alt.LayerChart:
    counts = scored.groupby(["risk_score", "risk_band"]).size().reset_index(name="applications")
    bars = (
        alt.Chart(counts)
        .mark_bar(width=5, cornerRadiusEnd=2)
        .encode(
            x=alt.X("risk_score:Q", title="Risk score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("applications:Q", title="Applications"),
            color=_band_color(),
            tooltip=[
                alt.Tooltip("risk_score:Q", title="Score"),
                alt.Tooltip("risk_band:N", title="Band"),
                alt.Tooltip("applications:Q", title="Applications", format=","),
            ],
        )
    )
    cutoffs = pd.DataFrame({"x": [low_max + 0.5, medium_max + 0.5]})
    rules = (
        alt.Chart(cutoffs)
        .mark_rule(strokeDash=[4, 4], color=PALETTES["light"]["muted"])
        .encode(x="x:Q")
    )
    return alt.layer(bars, rules).properties(height=280)


def default_rate_by_band(table: pd.DataFrame) -> alt.LayerChart:
    data = table.reset_index(names="risk_band").dropna(subset=["default_rate"])
    data["label"] = (data["default_rate"] * 100).round(1).astype(str) + "%"
    x = alt.X("risk_band:N", sort=list(BANDS), title=None, axis=alt.Axis(labelAngle=0))
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=56)
        .encode(
            x=x,
            y=alt.Y("default_rate:Q", title="Observed default rate", axis=alt.Axis(format="%")),
            color=_band_color().legend(None),
            tooltip=[
                alt.Tooltip("risk_band:N", title="Band"),
                alt.Tooltip("applications:Q", title="Applications", format=","),
                alt.Tooltip("defaults:Q", title="Defaults", format=","),
                alt.Tooltip("default_rate:Q", title="Default rate", format=".1%"),
            ],
        )
    )
    text = (
        alt.Chart(data)
        .mark_text(dy=-8, fontSize=12)
        .encode(x=x, y="default_rate:Q", text="label:N")
    )
    return alt.layer(bars, text).properties(height=260)


def roc_curves(test_scores: pd.DataFrame, labels: dict, mode: str = "light") -> alt.LayerChart:
    """labels maps a test_scores column (e.g. "scorecard") to its legend text."""
    colors = _palette(mode)
    frames = []
    for column, label in labels.items():
        points = roc_points(test_scores["defaulted"], test_scores[column])
        frames.append(points.assign(model=label))
    data = pd.concat(frames, ignore_index=True)

    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2, interpolate="step-after")
        .encode(
            x=alt.X(
                "false_positive_rate:Q",
                title="Share of repayers flagged",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "true_positive_rate:Q",
                title="Share of defaulters caught",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(labels.values()), range=list(colors["series"])),
                legend=alt.Legend(title=None, orient="bottom-right"),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("false_positive_rate:Q", title="Repayers flagged", format=".1%"),
                alt.Tooltip("true_positive_rate:Q", title="Defaulters caught", format=".1%"),
            ],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color=colors["muted"], strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )
    return alt.layer(diagonal, lines).properties(height=320)


def weights_comparison(
    weights: pd.DataFrame, current_points: dict, mode: str = "light"
) -> alt.Chart:
    colors = _palette(mode)
    rows = []
    for key, row in weights.iterrows():
        rows.append({"factor": row["label"], "policy": "Current", "points": current_points[key]})
        rows.append(
            {"factor": row["label"], "policy": "Suggested", "points": row["suggested_points"]}
        )
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=10)
        .encode(
            x=alt.X("points:Q", title="Max risk points"),
            y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("policy:N", sort=["Current", "Suggested"]),
            color=alt.Color(
                "policy:N",
                scale=alt.Scale(domain=["Current", "Suggested"], range=list(colors["series"])),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("factor:N", title="Factor"),
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip("points:Q", title="Max points"),
            ],
        )
        .properties(height=240)
    )
