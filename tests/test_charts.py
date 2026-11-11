import pandas as pd
import pytest

from risk_simulator import charts
from risk_simulator.data import generate_portfolio
from risk_simulator.pipeline import score_portfolio
from risk_simulator.scoring import DEFAULT_SCORECARD, score_application
from risk_simulator.validation import band_table, benchmark


@pytest.fixture(scope="module")
def scored():
    return score_portfolio(generate_portfolio(2_000, seed=9))


@pytest.fixture(scope="module")
def bench():
    return benchmark(generate_portfolio(2_000, seed=9))


def _datasets(chart):
    return list(chart.to_dict()["datasets"].values())


def test_score_meter_zones_follow_cutoffs():
    spec = charts.score_meter(70, 34, 64).to_dict()
    zones = next(d for d in spec["datasets"].values() if d and "start" in d[0])
    assert [(z["start"], z["end"]) for z in zones] == [(0, 35), (35, 65), (65, 101)]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_factor_contributions_one_row_per_factor(applicant, mode):
    chart = charts.factor_contributions(score_application(applicant(credit_score=500)), mode)
    rows = _datasets(chart)[0]
    assert len(rows) == len(DEFAULT_SCORECARD.factors)
    assert {"factor": "Credit score", "points": 25}.items() <= next(
        r for r in rows if r["factor"] == "Credit score"
    ).items()


def test_band_colours_are_the_status_palette(scored):
    spec = charts.score_distribution(scored, 34, 64).to_dict()
    scale = spec["layer"][0]["encoding"]["color"]["scale"]
    assert scale["domain"] == ["Low", "Medium", "High"]
    assert scale["range"] == ["#0ca30c", "#fab219", "#d03b3b"]


def test_score_distribution_counts_every_application(scored):
    counts = _datasets(charts.score_distribution(scored, 34, 64))
    bars = next(d for d in counts if d and "applications" in d[0])
    assert sum(row["applications"] for row in bars) == len(scored)


def test_default_rate_chart_skips_empty_bands():
    table = band_table(pd.DataFrame({"risk_band": ["Low", "Medium"], "defaulted": [0, 1]}))
    rows = _datasets(charts.default_rate_by_band(table))[0]
    assert [r["risk_band"] for r in rows] == ["Low", "Medium"]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_roc_curves_draw_both_models(bench, mode):
    labels = {"scorecard": "Scorecard", "logistic": "Logistic regression"}
    spec = charts.roc_curves(bench.test_scores, labels, mode).to_dict()
    color = spec["layer"][1]["encoding"]["color"]["scale"]
    assert color["domain"] == ["Scorecard", "Logistic regression"]
    assert color["range"] == list(charts.PALETTES[mode]["series"])


def test_weights_comparison_has_current_and_suggested(bench):
    current = {f.key: f.max_points for f in DEFAULT_SCORECARD.factors}
    rows = _datasets(charts.weights_comparison(bench.weights, current))[0]
    assert len(rows) == 2 * len(current)
    assert {r["policy"] for r in rows} == {"Current", "Suggested"}
