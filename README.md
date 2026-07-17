# Risk Scoring API

A small, general-purpose "upload your data, get a validated risk model" service.
Point it at any tabular dataset with a binary outcome column — churn, default,
fraud, no-show, anything — and it trains a logistic regression classifier,
validates it on a held-out split, and hands you back an API to score new
records against.

Built as a **reusable tool for future client work**, not a one-off demo tied
to a single dataset. The `/train` endpoint works on any CSV you give it.

Ships with one pre-trained model (`7cadb24d`, trained on a 5,000-row churn
sample) so there's something to query immediately after deployment, without
needing to train anything first.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/train` | Upload a CSV, get back a trained + validated model |
| GET | `/models` | List all trained models |
| GET | `/models/{model_id}` | Full metadata + validation report for one model |
| DELETE | `/models/{model_id}` | Remove a model |
| POST | `/predict/{model_id}` | Score new records against a trained model |
| GET | `/docs` | Interactive API documentation (auto-generated) |

## Repo setup

A few things matter here that a generic "push and deploy" glosses over:

- **The `Dockerfile` must sit at the repo root**, not nested in a subfolder.
  When you unzip and push, push the *contents* of this folder as the repo
  root — don't push a parent folder that contains this folder one level
  down, or Render won't find the Dockerfile.
- **`.gitignore`** is included and excludes caches, logs, and local env
  files. It does *not* exclude `models_store/` — the seeded demo model in
  there is meant to be committed, so it ships with every deploy.
- **`render.yaml`** is included as an optional Blueprint config. You don't
  need it for the manual "New Web Service" flow in the deployment steps
  above, but if you ever want a one-click reproducible deploy, Render can
  read this file directly (New → Blueprint, point it at the repo) instead
  of clicking through the individual settings.
- **`PORT`**: Render sets this automatically at runtime; the Dockerfile
  reads it via `${PORT}` with an 8000 fallback for local `docker run`, so
  no manual environment variable setup is needed on Render's side.

### Important limitation: storage is not persistent on the free tier

Render's free web services use an **ephemeral filesystem** — anything
written while the app is running (including new models trained via
`POST /train` after deployment) is lost on every restart, redeploy, or
free-tier spin-down/spin-up cycle. The seeded demo model survives because
it's baked into the Docker image at build time, but a client's own trained
model will not persist between sessions unless you either:

1. Attach a Render **persistent Disk** to the service (small paid add-on,
   a few dollars a month), or
2. Swap local file storage for something external (S3, a small Postgres
   instance) — a genuine "next iteration" if this tool sees real use.

Worth stating this plainly to any client who tries `/train` on the free
deployment and is surprised their model disappeared later — it's a known,
disclosed limitation of the demo tier, not a bug.

## Try it locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive documentation, or:

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

## Deploying it for real

This was built and fully tested locally, but actually putting it on the
public internet needs your own hosting account — I can't do that step from
here. Two straightforward options:

### Option A — Render (simplest, free tier, no AWS account needed)
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com), New → Web Service → connect the repo.
3. Render auto-detects the `Dockerfile`. Leave build/start commands blank.
4. Deploy. You'll get a public URL like `https://your-app.onrender.com` in a few minutes.

### Option B — AWS App Runner (matches the AWS skill on your CV)
1. Push this folder to a GitHub repo (or push the Docker image to ECR).
2. AWS Console → App Runner → Create service → source: your repo or ECR image.
3. App Runner reads the `Dockerfile` automatically; set the port to 8000.
4. Deploy. You get a public `*.awsapprunner.com` URL.

Either way, once it's live, `/docs` on the public URL is itself a good thing
to link a prospective client to — it's a working, interactive demonstration
of the service, not a screenshot.

## Design notes (for your own reference)

- **Model choice:** logistic regression, deliberately. It trains in seconds
  even on six-figure datasets, is well-calibrated out of the box, and is
  easy to explain to a non-technical client — that matters more here than
  squeezing out marginal AUC with a heavier model.
- **Storage:** trained models are saved to `models_store/` as `.joblib` files
  with a matching `.json` metadata file. This is fine for a demo/portfolio
  tool and for light real use, but a genuinely production deployment serving
  many clients would want a proper object store (S3) and a small database
  instead of the local filesystem — worth saying so upfront if a client asks.
- **Validation is not optional:** every `/train` call reports held-out AUC,
  precision/recall at a top-15% threshold, Brier score, and a calibration
  table — the same validation discipline as the churn case study, just
  generalised to work on anyone's data.
- **CORS is wide open** (`allow_origins=["*"]`) for demo convenience. A real
  client deployment should restrict this to their own domain.
