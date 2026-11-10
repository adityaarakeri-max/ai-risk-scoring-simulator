"""How well does the scorecard separate applicants who defaulted from those who didn't?

Standard credit-risk checks:
- AUC: chance a random defaulter scores higher than a random non-defaulter.
- Gini: 2 * AUC - 1, the figure most lenders quote (0 = no better than random, 1 = perfect).
- KS: the largest gap between the score distributions of defaulters and non-defaulters.
- Default rate by band: should rise from Low to Medium to High ("rank-ordering").
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .scoring import BANDS


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
