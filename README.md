# AI Risk-Scoring Simulator

A small end-to-end demo of how AI can support credit-risk and fraud decisioning:
rule-based checks and a weighted scoring model flag risky loan applications,
then the Claude API generates a plain-English explanation for every High-risk
case — the kind of summary a non-technical stakeholder (loan officer,
compliance reviewer) could actually read and act on.

This is a **portfolio project**, not a production system. Data is synthetic;
no real financial or personal records are used.

## Why this project

Built to demonstrate applied AI + risk-analysis skills for AI-adjacent fintech
roles (AI Operations Associate, AI Risk Analyst, Implementation Analyst). It
mirrors a simplified version of how automated scoring + LLM-generated
explanations show up in real credit scoring and fraud detection pipelines.

## Architecture

Three layers, each in its own module:

1. **Rule-based layer** ([src/scoring.py](src/scoring.py)) — five independent
   checks against simple thresholds (e.g. "is the loan-to-income ratio above
   50%?").
2. **Scoring layer** ([src/scoring.py](src/scoring.py)) — each rule
   contributes weighted points; the total (0-100) maps to a **Low / Medium /
   High** risk band.
3. **AI explanation layer** ([src/explain.py](src/explain.py)) — for every
   High-risk application, calls the Claude API with the triggered risk
   factors and asks for a 1-2 sentence plain-English explanation.

```
data/generate_data.py  →  synthetic_loans.csv
                               │
                               ▼
                    src/scoring.py (rules + points)
                               │
                               ▼
                 src/explain.py (Claude API, High-risk only)
                               │
                               ▼
                output/risk_scored_results.csv
```

## Risk factors and weights

| Factor | Max points | Signal |
|---|---|---|
| Loan-to-income ratio | 25 | Loan size relative to annual income |
| Credit score | 25 | Standard credit score band |
| Employment history | 20 | Years in current employment |
| Debt-to-income ratio | 20 | Existing monthly debt relative to income |
| Delinquencies (2yr) | 10 | Count of recent delinquencies |

Score bands: **Low** 0-34, **Medium** 35-64, **High** 65-100.

Thresholds and weights are illustrative, loosely modeled on common
underwriting heuristics — not calibrated against real default data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

Without an API key, the AI explanation layer falls back to a templated
summary built from the triggered rules, so the pipeline still runs end to end.

## Run

```bash
python data/generate_data.py   # creates data/synthetic_loans.csv
cd src && python main.py       # writes output/risk_scored_results.csv
```

## Output

`output/risk_scored_results.csv` — one row per applicant, with the computed
score, risk band, and (for High-risk rows) the AI-generated explanation.

## Possible extensions

- Swap synthetic data for a public anonymized credit-risk dataset
- Add a simple dashboard (e.g. Streamlit) over the output CSV
- Calibrate weights against a labeled default dataset instead of heuristics
