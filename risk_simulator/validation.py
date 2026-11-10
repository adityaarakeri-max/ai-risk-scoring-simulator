"""How well does the scorecard separate applicants who defaulted from those who didn't?

Standard credit-risk checks:
- AUC: chance a random defaulter scores higher than a random non-defaulter.
- Gini: 2 * AUC - 1, the figure most lenders quote (0 = no better than random, 1 = perfect).
- KS: the largest gap between the score distributions of defaulters and non-defaulters.
- Default rate by band: should rise from Low to Medium to High ("rank-ordering").
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import score_portfolio
from .scoring import BANDS, DEFAULT_SCORECARD, Scorecard


@dataclass(frozen=True)
class Discrimination:
    auc: float
    gini: float
    ks: float


def _require_both_outcomes(defaulted) -> None:
    if pd.Series(defaulted).nunique() < 2:
        raise ValueError("Need both defaulted and non-defaulted applicants to measure this.")


def discrimination(defaulted, score) -> Discrimination:
    _require_both_outcomes(defaulted)
    auc = float(roc_auc_score(defaulted, score))
    fpr, tpr, _ = roc_curve(defaulted, score)
    return Discrimination(auc=auc, gini=2 * auc - 1, ks=float((tpr - fpr).max()))


def roc_points(defaulted, score) -> pd.DataFrame:
    _require_both_outcomes(defaulted)
    fpr, tpr, _ = roc_curve(defaulted, score)
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})


def band_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Applications, share and default rate per risk band, in Low/Medium/High order."""
    grouped = scored.groupby("risk_band")["defaulted"].agg(applications="size", defaults="sum")
    table = grouped.reindex(BANDS, fill_value=0)
    table["share"] = table["applications"] / table["applications"].sum()
    table["default_rate"] = (table["defaults"] / table["applications"]).where(
        table["applications"] > 0
    )
    return table


def is_rank_ordered(table: pd.DataFrame) -> bool:
    rates = table["default_rate"].dropna()
    return bool(rates.is_monotonic_increasing and rates.is_unique)


@dataclass(frozen=True)
class Benchmark:
    scorecard: Discrimination
    reweighted: Discrimination
    logistic: Discrimination
    weights: pd.DataFrame
    suggested_weights: dict
    test_size: int


def _largest_remainder(shares: pd.Series, total: int) -> pd.Series:
    raw = shares * total
    rounded = raw.astype(int)
    shortfall = total - rounded.sum()
    for key in (raw - rounded).sort_values(ascending=False).index[:shortfall]:
        rounded[key] += 1
    return rounded


def benchmark(
    df: pd.DataFrame,
    scorecard: Scorecard = DEFAULT_SCORECARD,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Benchmark:
    """Compare the scorecard with a logistic regression fitted to the same five factors.

    The logistic model is a challenger, not a replacement: it shows how much
    discrimination the factors could deliver and which ones the scorecard
    over- or under-weights. Suggested weights are each factor's share of the
    model's standardised coefficients, rescaled to 100 points.
    """
    _require_both_outcomes(df["defaulted"])
    features = pd.DataFrame({f.key: f.measure(df) for f in scorecard.factors})
    x_train, x_test, y_train, y_test, _, df_test = train_test_split(
        features,
        df["defaulted"],
        df,
        test_size=test_fraction,
        stratify=df["defaulted"],
        random_state=seed,
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    coefs = pd.Series(model[-1].coef_[0], index=features.columns)

    shares = coefs.abs() / coefs.abs().sum()
    suggested = _largest_remainder(shares, 100)
    reweighted = scorecard.with_weights(suggested.to_dict())

    weights = pd.DataFrame(
        {
            "label": [f.label for f in scorecard.factors],
            "scorecard_share": [f.max_points / scorecard.max_points for f in scorecard.factors],
            "model_share": shares.values,
            "suggested_points": suggested.values,
            "direction_agrees": [
                (coefs[f.key] > 0) == f.higher_is_riskier for f in scorecard.factors
            ],
        },
        index=features.columns,
    )

    return Benchmark(
        scorecard=discrimination(y_test, score_portfolio(df_test, scorecard)["risk_score"]),
        reweighted=discrimination(y_test, score_portfolio(df_test, reweighted)["risk_score"]),
        logistic=discrimination(y_test, model.predict_proba(x_test)[:, 1]),
        weights=weights,
        suggested_weights={k: int(v) for k, v in suggested.items()},
        test_size=len(y_test),
    )
