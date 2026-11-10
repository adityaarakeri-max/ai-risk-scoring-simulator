import pandas as pd
import pytest

from risk_simulator.data import generate_portfolio
from risk_simulator.pipeline import score_portfolio
from risk_simulator.validation import band_table, discrimination, is_rank_ordered, roc_points


def test_perfect_separation():
    result = discrimination([0, 0, 1, 1], [10, 20, 80, 90])
    assert result.auc == 1.0
    assert result.gini == 1.0
    assert result.ks == 1.0


def test_reversed_scores_have_negative_gini():
    result = discrimination([0, 0, 1, 1], [90, 80, 20, 10])
    assert result.gini == -1.0


def test_uninformative_score_has_zero_gini():
    result = discrimination([0, 1, 0, 1], [50, 50, 50, 50])
    assert result.auc == 0.5
    assert result.gini == 0.0


def test_one_outcome_only_is_rejected():
    with pytest.raises(ValueError, match="both defaulted and non-defaulted"):
        discrimination([0, 0, 0], [1, 2, 3])


def test_roc_points_run_from_corner_to_corner():
    points = roc_points([0, 1, 0, 1], [1, 2, 3, 4])
    assert points.iloc[0].tolist() == [0.0, 0.0]
    assert points.iloc[-1].tolist() == [1.0, 1.0]


def test_band_table_orders_bands_and_fills_empty_ones():
    scored = pd.DataFrame(
        {"risk_band": ["Low", "Low", "Medium", "Medium"], "defaulted": [0, 1, 1, 1]}
    )
    table = band_table(scored)
    assert list(table.index) == ["Low", "Medium", "High"]
    assert table.loc["Low", "default_rate"] == 0.5
    assert table.loc["High", "applications"] == 0
    assert pd.isna(table.loc["High", "default_rate"])
    assert table["share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ([0.05, 0.2, 0.5], True),
        ([0.05, 0.2, None], True),
        ([0.2, 0.1, 0.5], False),
        ([0.1, 0.1, 0.3], False),
    ],
)
def test_rank_ordering(rates, expected):
    table = pd.DataFrame({"default_rate": rates}, index=["Low", "Medium", "High"])
    assert is_rank_ordered(table) is expected


def test_default_scorecard_rank_orders_simulated_portfolio():
    scored = score_portfolio(generate_portfolio(10_000, seed=3))
    assert is_rank_ordered(band_table(scored))
    result = discrimination(scored["defaulted"], scored["risk_score"])
    assert 0.3 < result.gini < 0.7
