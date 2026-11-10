"""Interactive front end. Run with: streamlit run streamlit_app.py"""

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from risk_simulator import charts, data, explain, pipeline, validation
from risk_simulator.scoring import BANDS, DEFAULT_SCORECARD, Scorecard

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_applications.csv"

SESSION_AI_LIMIT = 10
DAILY_AI_LIMIT = 200
DEFAULT_WEIGHTS = {f.key: f.max_points for f in DEFAULT_SCORECARD.factors}
DEFAULT_CUTOFFS = (DEFAULT_SCORECARD.low_max, DEFAULT_SCORECARD.medium_max)
BAND_BADGE = {
    "Low": ("green", ":material/check_circle:"),
    "Medium": ("orange", ":material/warning:"),
    "High": ("red", ":material/error:"),
}
EXAMPLES = {
    "Stretched": {
        "annual_income": 32_000,
        "loan_amount": 24_000,
        "credit_score": 590,
        "employment_years": 1.5,
        "monthly_debt_payments": 1_150,
        "delinquencies_2yr": 2,
    },
    "Typical": {
        "annual_income": 42_000,
        "loan_amount": 9_000,
        "credit_score": 690,
        "employment_years": 5.0,
        "monthly_debt_payments": 700,
        "delinquencies_2yr": 0,
    },
    "Strong": {
        "annual_income": 85_000,
        "loan_amount": 10_000,
        "credit_score": 790,
        "employment_years": 12.0,
        "monthly_debt_payments": 600,
        "delinquencies_2yr": 0,
    },
}


def configure_api_key() -> None:
    load_dotenv()
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except StreamlitSecretNotFoundError:
        key = None
    if key:
        os.environ.setdefault("ANTHROPIC_API_KEY", key)


@st.cache_resource
def daily_ai_usage() -> dict:
    return {"date": dt.date.today(), "count": 0}


def ai_budget_left() -> int:
    usage = daily_ai_usage()
    if usage["date"] != dt.date.today():
        usage.update(date=dt.date.today(), count=0)
    session_left = SESSION_AI_LIMIT - st.session_state.ai_calls
    return max(0, min(session_left, DAILY_AI_LIMIT - usage["count"]))


def record_ai_call() -> None:
    st.session_state.ai_calls += 1
    daily_ai_usage()["count"] += 1


def init_state() -> None:
    for key, points in DEFAULT_WEIGHTS.items():
        st.session_state.setdefault(f"w_{key}", points)
    st.session_state.setdefault("cutoffs", DEFAULT_CUTOFFS)
    st.session_state.setdefault("ai_calls", 0)
    st.session_state.setdefault("ai_texts", {})
    if "in_annual_income" not in st.session_state:
        load_example("Stretched")


def load_example(name: str) -> None:
    for field, value in EXAMPLES[name].items():
        st.session_state[f"in_{field}"] = value


def reset_policy() -> None:
    for key, points in DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{key}"] = points
    st.session_state.cutoffs = DEFAULT_CUTOFFS


def build_scorecard(weights: tuple, low_max: int, medium_max: int) -> Scorecard:
    weighted = DEFAULT_SCORECARD.with_weights(dict(weights))
    return Scorecard(factors=weighted.factors, low_max=low_max, medium_max=medium_max)


def policy_sidebar() -> tuple:
    with st.sidebar:
        st.header("Scorecard policy")
        st.caption(
            "Set the most risk points each factor can add. "
            "Scores are rescaled to 0–100, so the band cutoffs keep their meaning."
        )
        for factor in DEFAULT_SCORECARD.factors:
            st.slider(
                factor.label,
                0,
                50,
                key=f"w_{factor.key}",
                help=f"{factor.reason_code}: {factor.reason}",
            )
        low_max, medium_max = st.slider(
            "Band cutoffs",
            0,
            99,
            key="cutoffs",
            help="Scores up to the first handle are Low, up to the second are Medium, "
            "and anything above is High.",
        )
        st.button("Reset to default policy", on_click=reset_policy, width="stretch")

    weights = tuple((key, st.session_state[f"w_{key}"]) for key in DEFAULT_WEIGHTS)
    if sum(points for _, points in weights) == 0:
        st.error("Give at least one factor some points in the sidebar.")
        st.stop()
    if low_max >= medium_max:
        st.error("The Low cutoff must be below the Medium cutoff. Adjust it in the sidebar.")
        st.stop()
    return weights, low_max, medium_max


def applicant_inputs() -> dict:
    st.segmented_control(
        "Start from an example",
        list(EXAMPLES),
        key="example",
        on_change=lambda: st.session_state.example and load_example(st.session_state.example),
    )
    left, right = st.columns(2)
    with left:
        st.number_input("Annual income (£)", 1_000, 1_000_000, step=1_000, key="in_annual_income")
        st.number_input("Loan amount (£)", 0, 500_000, step=500, key="in_loan_amount")
        st.number_input(
            "Monthly debt payments (£)",
            0,
            50_000,
            step=50,
            key="in_monthly_debt_payments",
            help="Existing repayments on cards, car finance and other loans.",
        )
    with right:
        st.slider("Credit score", 300, 850, key="in_credit_score")
        st.slider(
            "Years in current job", 0.0, 40.0, step=0.5, format="%.1f", key="in_employment_years"
        )
        st.slider("Missed payments in the last 2 years", 0, 8, key="in_delinquencies_2yr")
    return {field: st.session_state[f"in_{field}"] for field in EXAMPLES["Typical"]}


def explanation_panel(scored, key: str) -> None:
    st.subheader("Explanation")
    payload_key = json.dumps(explain.explanation_payload(scored), sort_keys=True)
    ai_text = st.session_state.ai_texts.get(payload_key)

    if ai_text:
        st.info(ai_text, icon=":material/auto_awesome:")
        st.caption(
            f"Written by Claude ({explain.model_name()}) from the reason codes above. "
            "The score and band come from the scorecard, not from the AI."
        )
        return

    st.write(explain.template_explanation(scored))
    if not explain.ai_available():
        st.caption("Template explanation. Claude explanations are off because no API key is set.")
        return

    budget = ai_budget_left()
    if st.button(
        "Explain with Claude",
        key=f"explain_{key}",
        icon=":material/auto_awesome:",
        disabled=budget == 0,
        help="Claude rewrites the reasons in plain English. It can't change the score.",
    ):
        with st.spinner("Asking Claude…"):
            result = explain.explain(scored)
        if result.source == "ai":
            record_ai_call()
            st.session_state.ai_texts[payload_key] = result.text
            st.rerun()
        st.warning(f"Showing the template instead: {result.note}.")
    st.caption(
        "Template explanation. "
        + (
            f"{budget} Claude explanations left in this session."
            if budget
            else "The Claude explanation limit for this session or today has been reached."
        )
    )


def applicant_tab(scorecard: Scorecard, mode: str) -> None:
    inputs_col, result_col = st.columns([5, 6], gap="large")
    with inputs_col:
        record = applicant_inputs()
    scored = scorecard.score(record)

    with result_col:
        score_col, band_col = st.columns([1, 2], vertical_alignment="bottom")
        score_col.metric("Risk score", f"{scored.score} / 100")
        color, icon = BAND_BADGE[scored.risk_band]
        band_col.badge(f"{scored.risk_band} risk", icon=icon, color=color)
        st.altair_chart(
            charts.score_meter(scored.score, scorecard.low_max, scorecard.medium_max, mode),
            width="stretch",
        )

        reasons = scored.principal_reasons()
        st.subheader("Principal reasons")
        if reasons:
            for r in reasons:
                st.markdown(
                    f"`{r.reason_code}` {r.reason} — {r.label.lower()} {r.display_value} "
                    f"(+{r.points} pts)"
                )
        else:
            st.write("No risk factors were triggered.")

        explanation_panel(scored, key="applicant")

    st.subheader("Where the points came from")
    st.altair_chart(charts.factor_contributions(scored, mode), width="stretch")
    with st.expander("Show as a table"):
        st.dataframe(
            [
                {
                    "Factor": f.label,
                    "Value": f.display_value,
                    "Points": f.points,
                    "Max points": f.max_points,
                    "Reason code": f.reason_code,
                }
                for f in scored.factors
            ],
            hide_index=True,
        )


@st.cache_data
def sample_portfolio() -> pd.DataFrame:
    return data.load_applications(SAMPLE_PATH)


@st.cache_data
def simulated_portfolio(n: int, seed: int) -> pd.DataFrame:
    return data.generate_portfolio(n=n, seed=seed)


@st.cache_data
def scored_portfolio(df: pd.DataFrame, weights: tuple, low_max: int, medium_max: int):
    return pipeline.score_portfolio(df, build_scorecard(weights, low_max, medium_max))


def template_csv() -> bytes:
    return sample_portfolio().drop(columns=["defaulted"]).head(5).to_csv(index=False).encode()


def portfolio_source() -> tuple:
    """Returns (applications DataFrame or None, a short description of the source)."""
    choice = st.segmented_control(
        "Applications",
        ["Sample file", "Simulate a portfolio", "Upload a CSV"],
        default="Sample file",
        key="source",
    )
    if choice == "Simulate a portfolio":
        size_col, seed_col = st.columns([3, 1])
        n = size_col.slider("Number of applications", 500, 20_000, 5_000, step=500, key="sim_n")
        seed = seed_col.number_input("Random seed", 0, 9_999, 42, key="sim_seed")
        return simulated_portfolio(n, seed), f"{n:,} simulated applications (seed {seed})"

    if choice == "Upload a CSV":
        st.download_button(
            "Download a template",
            template_csv(),
            file_name="applications_template.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        upload = st.file_uploader(
            "CSV with annual_income, loan_amount, credit_score, employment_years, "
            "monthly_debt_payments and delinquencies_2yr. Add a 0/1 'defaulted' column "
            "to unlock validation.",
            type="csv",
            key="upload",
        )
        if upload is None:
            return None, ""
        try:
            return data.load_applications(upload), upload.name
        except data.DataValidationError as exc:
            st.error(f"{upload.name}: {exc}")
            return None, ""

    return sample_portfolio(), "the 1,000-application sample file"


def portfolio_tab(weights: tuple, low_max: int, medium_max: int) -> None:
    df, source = portfolio_source()
    st.session_state.portfolio = (df, source)
    if df is None:
        st.info("Upload a CSV to score it, or pick the sample file.", icon=":material/upload:")
        return

    scorecard = build_scorecard(weights, low_max, medium_max)
    scored = scored_portfolio(df, weights, low_max, medium_max)
    has_outcomes = "defaulted" in scored.columns
    counts = scored["risk_band"].value_counts()

    kpis = st.columns(4)
    kpis[0].metric("Applications", f"{len(scored):,}")
    kpis[1].metric("Rated High risk", f"{counts.get('High', 0) / len(scored):.1%}")
    kpis[2].metric("Average score", f"{scored['risk_score'].mean():.1f}")
    if has_outcomes:
        kpis[3].metric("Observed default rate", f"{scored['defaulted'].mean():.1%}")
    else:
        kpis[3].metric("Observed default rate", "Not in file")
    st.caption(f"Scoring {source} with the policy in the sidebar.")

    chart_col, table_col = st.columns([3, 2], gap="large")
    with chart_col:
        st.subheader("Score distribution")
        st.altair_chart(charts.score_distribution(scored, low_max, medium_max), width="stretch")
    with table_col:
        st.subheader("By band")
        summary = pd.DataFrame(
            {
                "Applications": [int(counts.get(b, 0)) for b in BANDS],
                "Share": [counts.get(b, 0) / len(scored) for b in BANDS],
            },
            index=list(BANDS),
        )
        if has_outcomes:
            summary["Default rate"] = validation.band_table(scored)["default_rate"]
        st.dataframe(
            summary,
            column_config={
                "Share": st.column_config.NumberColumn(format="percent"),
                "Default rate": st.column_config.NumberColumn(format="percent"),
            },
        )

    st.subheader("Applications")
    bands = st.pills(
        "Show bands", list(BANDS), selection_mode="multi", default=["High"], key="band_filter"
    )
    shown = scored[scored["risk_band"].isin(bands or BANDS)]
    columns = [
        "applicant_id",
        "risk_score",
        "risk_band",
        "reason_codes",
        "annual_income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "monthly_debt_payments",
        "delinquencies_2yr",
    ] + (["defaulted"] if has_outcomes else [])
    st.dataframe(
        shown[columns].sort_values("risk_score", ascending=False),
        hide_index=True,
        height=300,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%d"
            ),
            "reason_codes": st.column_config.TextColumn("Reason codes"),
            "applicant_id": st.column_config.NumberColumn("ID", format="%d"),
            "risk_band": "Band",
            "annual_income": st.column_config.NumberColumn("Income", format="localized"),
            "loan_amount": st.column_config.NumberColumn("Loan", format="localized"),
            "credit_score": "Credit score",
            "employment_years": st.column_config.NumberColumn("Years employed", format="%.1f"),
            "monthly_debt_payments": st.column_config.NumberColumn(
                "Monthly debt", format="localized"
            ),
            "delinquencies_2yr": "Missed payments",
            "defaulted": st.column_config.CheckboxColumn("Defaulted"),
        },
    )
    st.download_button(
        "Download scored results",
        scored.to_csv(index=False).encode(),
        file_name="risk_scored_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    if shown.empty:
        return
    st.subheader("Inspect an application")
    ordered = shown.sort_values("risk_score", ascending=False)
    applicant_id = st.selectbox(
        "Applicant",
        ordered["applicant_id"],
        format_func=lambda i: f"#{i}",
        key="inspect_id",
    )
    record = df[df["applicant_id"] == applicant_id].iloc[0].to_dict()
    scored_one = scorecard.score(record)
    reasons = scored_one.principal_reasons()
    st.markdown(
        f"**#{applicant_id}: {scored_one.score}/100, {scored_one.risk_band} risk.** "
        + (" ".join(f"`{r.reason_code}`" for r in reasons) or "No reason codes.")
    )
    explanation_panel(scored_one, key=f"portfolio_{applicant_id}")


def main() -> None:
    st.set_page_config(
        page_title="AI risk-scoring simulator",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    configure_api_key()
    init_state()

    st.title("AI risk-scoring simulator")
    st.caption(
        "A rules-based credit scorecard decides the risk band. "
        "Claude only explains the decision in plain English. All data is simulated."
    )

    weights, low_max, medium_max = policy_sidebar()
    scorecard = build_scorecard(weights, low_max, medium_max)
    mode = st.context.theme.type or "light"

    tab_applicant, tab_portfolio = st.tabs(["Score an applicant", "Score a portfolio"])
    with tab_applicant:
        applicant_tab(scorecard, mode)
    with tab_portfolio:
        portfolio_tab(weights, low_max, medium_max)


main()
