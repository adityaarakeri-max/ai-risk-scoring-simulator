"""Drive the Streamlit app headlessly, the way a visitor would."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "streamlit_app.py")
FACTORS = [
    "loan_to_income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "delinquencies_2yr",
]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    return AppTest.from_file(APP, default_timeout=120).run()


def metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


def test_app_loads_with_example_applicant(app):
    assert not app.exception
    assert metric(app, "Risk score") == "75 / 100"
    assert any("High risk" in m.value for m in app.markdown)


def test_changing_inputs_rescores_live(app):
    app.slider(key="in_credit_score").set_value(800).run()
    assert metric(app, "Risk score") == "60 / 100"


def test_reweighting_changes_the_score(app):
    app.slider(key="w_employment_years").set_value(0).run()
    # 65 of the remaining 80 points -> 81
    assert metric(app, "Risk score") == "81 / 100"


def test_invalid_cutoffs_show_an_error(app):
    app.slider(key="cutoffs").set_value((60, 60)).run()
    assert "Low cutoff must be below the Medium cutoff" in app.error[0].value


def test_all_zero_weights_show_an_error(app):
    for key in FACTORS:
        app.slider(key=f"w_{key}").set_value(0)
    app.run()
    assert "at least one factor" in app.error[0].value


def test_template_explanation_without_api_key(app):
    assert not [b for b in app.button if b.label == "Explain with Claude"]
    assert any("no API key is set" in c.value for c in app.caption)


def test_portfolio_tab_scores_sample(app):
    assert metric(app, "Applications") == "1,000"
    assert metric(app, "Observed default rate") == "10.0%"


def test_validation_tab_reports_rank_ordering(app):
    assert metric(app, "Gini").startswith("0.")
    assert any("default rate rises by band" in m.value for m in app.markdown)


def test_apply_suggested_weights_updates_sidebar(app):
    next(b for b in app.button if b.label == "Apply suggested weights").click().run()
    weights = [app.session_state[f"w_{k}"] for k in FACTORS]
    assert weights != [25, 25, 20, 20, 10]
    assert sum(weights) == 100


def test_reset_restores_default_policy(app):
    app.slider(key="w_credit_score").set_value(60).run()
    next(b for b in app.button if b.label == "Reset to default policy").click().run()
    assert app.session_state["w_credit_score"] == 25
