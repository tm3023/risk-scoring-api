# Risk Scoring API

A small, general-purpose model-scoring service. Point it at any tabular
dataset with a binary outcome column (churn, default, fraud, no-show, or
anything else) and it trains a logistic regression classifier, validates it
on a held-out split and returns an API to score new records against.

Built as a reusable tool rather than a demo tied to a single dataset: the
`/train` endpoint works on any CSV supplied to it.

Ships with one pre-trained model (`7cadb24d`, trained on a 5,000-row churn
sample) so there is something to query immediately after deployment,
without needing to train anything first.

A working deployment is available at
[risk-scoring-api-1tct.onrender.com](https://risk-scoring-api-1tct.onrender.com),
serving three things from one place: a frontend demo of this API, the full
interactive churn analytics case study this project grew out of and the
formal written reports produced alongside it.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Frontend demo of this API |
| GET | `/dashboard` | Interactive churn analytics case study dashboard |
| GET | `/reports/Customer_Retention_Decision_Report.pdf` | Formal decision report |
| GET | `/reports/Project_Guide.pdf` | One-page guide to the whole project |
| GET | `/health` | Liveness check |
| GET | `/status` | Service metadata and endpoint list |
| POST | `/train` | Upload a CSV, get back a trained and validated model |
| GET | `/models` | List all trained models |
| GET | `/models/{model_id}` | Full metadata and validation report for one model |
| DELETE | `/models/{model_id}` | Remove a model |
| POST | `/predict/{model_id}` | Score new records against a trained model |
| GET | `/docs` | Interactive API documentation (auto-generated) |

## Design decisions

- **Model choice.** Logistic regression, deliberately. It trains in seconds
  even on six-figure datasets, is well calibrated out of the box and is
  straightforward to explain to a non-technical stakeholder. That matters
  more here than squeezing out marginal AUC with a heavier model.
- **Validation is not optional.** Every `/train` call reports held-out AUC,
  precision and recall at a top-15% threshold, a Brier score and a full
  calibration table, the same validation discipline applied throughout the
  accompanying churn case study, generalised to work on arbitrary data.
- **Storage.** Trained models are saved to `models_store/` as `.joblib`
  files with matching `.json` metadata. This is adequate for a demo or
  light real use. A production deployment serving multiple clients would
  use an external object store (S3) and a small database instead of the
  local filesystem; see the storage limitation below.
- **CORS is open** (`allow_origins=["*"]`) for demonstration purposes. A
  production deployment should restrict this to a specific origin.

## Known limitations

**Storage is not persistent on free-tier hosting.** Render's free web
services use an ephemeral filesystem: anything written while the app is
running, including new models trained via `POST /train` after deployment,
is lost on restart, redeploy or a free-tier spin-down and spin-up cycle.
The seeded demo model survives because it is baked into the Docker image
at build time, but a model trained live on the free deployment will not
persist between sessions. Fixing this requires either a persistent disk
attached to the service or external storage (S3, a small Postgres
instance).

**The churn classifier is a moderate discriminator** (AUC approximately
0.67 on held-out data). It is well suited to ranking relative risk and
producing calibrated probabilities, not intended for hard yes/no
decisions on individual records.

**Free-tier cold starts.** If the hosted deployment has been idle for more
than 15 minutes, the first request takes 30 to 50 seconds while the
instance wakes up.

## Repo structure

- `app/main.py`: FastAPI application and route definitions
- `app/ml.py`: training, validation and prediction logic
- `app/frontend.html`: the API demo page served at `/`
- `app/dashboard.html`: the churn analytics case study, served at `/dashboard`
- `reports/`: the formal decision report and project guide, served as
  static downloads under `/reports/`
- `models_store/`: persisted trained models (the seeded demo model ships
  here and is tracked in git; models trained afterward are not persisted
  on free-tier hosting, see above)
- `sample_data/`: a sample CSV for testing `/train`
- `Dockerfile`, `render.yaml`: deployment configuration

The `Dockerfile` must sit at the repository root. When pushing to a new
repository, push the contents of this folder as the repository root
rather than nesting it one level down, or a platform reading the
`Dockerfile` automatically will not find it.

`PORT` is set automatically by most hosting platforms at runtime; the
Dockerfile reads it via `${PORT}` with an 8000 fallback for local
`docker run`, so no manual environment variable configuration is required.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/` for the demo page, or `/docs` for
interactive API documentation.

```bash
# Train a model
curl -X POST http://127.0.0.1:8000/train \
  -F "file=@sample_data/demo_churn_sample.csv" \
  -F "target_column=churn" \
  -F "model_name=my_first_model"

# Score a new record against it
curl -X POST http://127.0.0.1:8000/predict/<model_id> \
  -H "Content-Type: application/json" \
  -d '{"records": [{"tenure": 3, "contract": "month_to_month", "customer_satisfaction": 2, "num_complaints": 4, "late_payments": 2, "monthlycharges": 95}]}'
```

## Deployment

Tested with Render (free tier, Dockerfile auto-detected) and compatible
with any platform that builds from a standard Dockerfile, including AWS
App Runner. `render.yaml` is included as an optional Blueprint
configuration for one-step reproducible deploys on Render; it is not
required for a manual setup through Render's dashboard.
