# Insurance Premium Predictor

A Streamlit dashboard built on top of a two-stage model trained on the
HealthWise dataset:

1. **Risk tier classifier** (`DecisionTreeClassifier`) predicts whether a
   customer is **Low**, **Medium**, or **High** risk from their profile
   (age, BMI, children, exercise frequency, sex, smoker status, region).
2. **Per-tier premium regressor** (`LinearRegression`, one model per tier)
   predicts the customer's `annual_charge` given their assigned tier.

## Project structure

```
insurance-premium-dashboard/
├── app.py                  # Streamlit dashboard (self-contained, no local imports)
├── train_model.py          # Data cleaning + model training (standalone CLI script)
├── generate_dataset.py     # Regenerates healthwise.csv from scratch (optional)
├── healthwise.csv          # Clean, ready-to-use dataset (1203 rows, no missing/duplicates)
├── healthwise_model.joblib # Pre-trained model artifact, loads instantly on startup
├── requirements.txt
└── README.md
```

## About healthwise.csv

This is a **synthetic, clean stand-in dataset** — not real customer
data — generated to match the schema and modeling relationships of the
original notebook (1203 rows, 17 columns, no missing values or
duplicates). Smoking status and age are the dominant drivers of
`annual_charge`, exercise frequency gives a modest discount, and several
columns (`favorite_color`, `zodiac_sign`, `lucky_number`,
`preferred_contact`, `marketing_opt_in`) are intentionally irrelevant
noise, mirroring the original data's feature audit.

Swap it out for your real dataset at any time — just keep the same
column names. To regenerate the synthetic version (e.g. with a
different random seed or row count), run:

```bash
python generate_dataset.py
```

## Setup

```bash
pip install -r requirements.txt
```

`healthwise.csv` and a pre-trained `healthwise_model.joblib` are already
included, so the dashboard will load instantly without any extra setup.
If you replace the CSV with your own data, delete
`healthwise_model.joblib` first so the app retrains on your data (or run
`python train_model.py` to regenerate it manually).

## Run the dashboard

```bash
streamlit run app.py
```

Since `healthwise.csv` and `healthwise_model.joblib` are already
included, this works immediately — no upload step needed. The upload
prompt only appears if both files are missing (e.g. if you deliberately
excluded the CSV from a deployment).

## Dashboard tabs

- **🔮 Predict** — enter a customer profile and get their predicted risk
  tier and annual premium, plus a "what-if" tool to see how changing one
  factor (e.g. smoking status) shifts the outcome.
- **📊 Data Insights** — distribution of charges, boxplots by smoker
  status and risk tier, and a raw data preview (requires `healthwise.csv`
  to be available).
- **📈 Model Performance** — held-out classifier accuracy and per-tier
  regression R² / MAE from training time.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app
   at `app.py`.
3. Either commit `healthwise.csv` to the repo (simplest) or leave it out
   and use the in-app upload prompt each session.

## Design note: app.py is self-contained

`app.py` does **not** import from `train_model.py` — all model training and
prediction logic is duplicated inline in `app.py` itself. This means the
dashboard can never fail with a `ModuleNotFoundError` for a missing sibling
file, regardless of repo layout or how the hosting platform resolves paths.
`train_model.py` is kept only as an optional, independent CLI script for
offline training/experimentation (`python train_model.py`) — you don't need
to deploy it for the dashboard to work.

If you do see a `ModuleNotFoundError`, it means a *listed* dependency from
`requirements.txt` isn't installed — double-check that file made it into the
repo and that your Streamlit Cloud app's "Manage app" logs show it being
installed at startup.
