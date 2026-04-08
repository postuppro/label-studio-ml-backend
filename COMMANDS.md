# Project Commands

This document collects the common commands and workflows used in this repository.

## Prerequisites

- Docker + Docker Compose (recommended for running the ML backend)
- Python (for local development)
- A running Label Studio instance

## Clone the repo

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
```

## Install (local development)

Install the package in editable mode:

```bash
pip install -e .
```

Install dependencies using the root Makefile:

```bash
make install
```

If you want to install the pinned dependencies directly:

```bash
pip install -r requirements.txt
```

## Run the ML backend (Docker)

### Start the example backend

From a model directory containing a `docker-compose.yml` (for example, `my_ml_backend/`):

```bash
docker compose up --build
```

The ML backend should be available on:

- `http://localhost:9090/health`
- `http://localhost:9090/predict`

### View logs

```bash
docker compose logs -f
```

### Stop

```bash
docker compose down
```

### Rebuild without cache

```bash
docker compose build --no-cache
```

## Run the ML backend (without Docker)

Start the server using the CLI entrypoint:

```bash
label-studio-ml start my_ml_backend
```

Change the port:

```bash
label-studio-ml start my_ml_backend -p 9091
```

## Environment variables

Common environment variables used by the ML backend:

- `LABEL_STUDIO_URL`
- `LABEL_STUDIO_API_KEY`
- `MODEL_DIR`
- `LOG_LEVEL`
- `BASIC_AUTH_USER`
- `BASIC_AUTH_PASS`
- `WORKERS`
- `THREADS`

In `my_ml_backend/docker-compose.yml` they are configured under `services.ml-backend.environment`.

If you're using the `my_ml_backend/docker-compose.yml` provided in this repo, note:

- `MODEL_DIR=/ml-backend-data/models`
- `./data/server:/ml-backend-data` (persisted backend state; includes `cache.db` under `${MODEL_DIR}`)
- `./training/runs:/ml-backend-data/models/runs` (Ultralytics training artifacts)
- `./training/datasets:/ml-backend-data/models/datasets` (generated YOLO datasets)

## Connect the ML backend to Label Studio

In Label Studio:

- Go to Project Settings
- Add an ML backend and point it at `http://<host>:9090`

## Trigger training

### Trigger training from Label Studio UI

- Project Settings
- Model
- Start Training

### Trigger training via Label Studio API

Label Studio provides an endpoint to trigger training for a connected ML backend:

```bash
curl -X POST http://localhost:8080/api/ml/{id}/train
```

Where `{id}` is the ML backend ID in Label Studio.

### Training logs

Training output is written to the ML backend process stdout/stderr, so you can view it via Docker logs:

```bash
docker compose logs -f
```

If training artifacts are configured to be written under `MODEL_DIR`, they will be persisted to the mounted volume.

With the `my_ml_backend` training implementation in this repo, Ultralytics writes run artifacts to:

- `${MODEL_DIR}/runs/...` inside the container

Given `MODEL_DIR=/ml-backend-data/models`, artifacts end up under:

- `/ml-backend-data/models/runs/...`

and are persisted on the host via the `./data/server:/ml-backend-data` volume.

### Training from exported `/data` (custom backend workflow)

If you are exporting per-frame annotations to the directory mounted at `/data` (for example `my_ml_backend/docker-compose.yml` mounts a host folder into `/data`), training expects:

- `/data/annotations/<file-id>/frame_*.json`
- `/data/frames/<file-id>/frame_*.jpg` (path referenced in each JSON under `data.image_fs`)

After exporting, trigger training through Label Studio:

```bash
curl -X POST http://localhost:8080/api/ml/{id}/train
```

## Health check

```bash
curl http://localhost:9090/health
```

## Predict

Send a prediction request (example skeleton):

```bash
curl -X POST http://localhost:9090/predict \
  -H 'Content-Type: application/json' \
  -d '{"tasks": [], "project": "<project_id>.<timestamp>", "label_config": "<xml>"}'
```

## Model approval workflow (custom backend)

The `my_ml_backend` example includes an optional human approval workflow for trained weights.

- Training saves `best.pt` to the runs directory on disk.
- Predictions use the **approved** model if one is set, otherwise the **base model** from `config.json`.
- You explicitly approve a trained model by providing its weights path.
- Approval state is persisted to `model_state.json` on the mounted volume and survives container restarts.

These endpoints are served by the ML backend on port `9090`.

Note:

- Label Studio sends the ML backend `project` in the format `"<project_id>.<timestamp>"`.
- This backend only uses the prefix before the dot as the effective project id.
- For local testing, `"<project_id>.0"` is sufficient.

### Get active model

```bash
curl -X POST http://localhost:9090/models/active \
  -H 'Content-Type: application/json' \
  -d '{"project": "<project_id>.0"}'
```

### Approve a model

Set a trained run as the active model by job ID. The backend resolves
`<TRAINING_RUNS_DIR>/<job_id>/weights/best.pt` and loads it immediately.

```bash
curl -X POST http://localhost:9090/models/approve \
  -H 'Content-Type: application/json' \
  -d '{"project": "<project_id>.0", "job_id": "<job_id>"}'
```

### Reset to base model

Clear the approved model and revert to the base model from `config.json`.

```bash
curl -X POST http://localhost:9090/models/reset \
  -H 'Content-Type: application/json' \
  -d '{"project": "<project_id>.0"}'
```

### CLI helper script

```bash
# Show the currently active model
python model_approval_cli.py --project-id <project_id> active

# Approve a trained model by job ID
python model_approval_cli.py --project-id <project_id> approve --job-id <job_id>

# Reset to base model
python model_approval_cli.py --project-id <project_id> reset
```

You can also resolve project id from a Label Studio project title (requires `label-studio-sdk`):

```bash
python model_approval_cli.py \
  --project-name "<project_title>" \
  --ls-url http://localhost:8080 \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  active
```

## Tests

Run tests:

```bash
make test
```

Or directly:

```bash
pytest tests
```

## Lint / format

This repository doesn’t define a single canonical lint/format command in the root Makefile.
If you add tooling (ruff/black/isort), document the exact commands here.
