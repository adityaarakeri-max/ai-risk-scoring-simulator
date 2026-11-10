"""Interactive front end. Run with: streamlit run streamlit_app.py"""

import datetime as dt
import json
import os

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from risk_simulator import charts, explain
from risk_simulator.scoring import DEFAULT_SCORECARD, Scorecard

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


def explanation_panel(scored) -> None:
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

        explanation_panel(scored)

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

    (tab_applicant,) = st.tabs(["Score an applicant"])
    with tab_applicant:
        applicant_tab(scorecard, mode)


main()
