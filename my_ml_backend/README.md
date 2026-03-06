This guide describes the simplest way to start using ML backend with Label Studio.

## Running with Docker (Recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Building from source (Advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./dir_with_your_model
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server

## Custom model changes in this repository

This `my_ml_backend` implementation has been customized to run prompt-based object detection using Ultralytics YOLOE-26 weights.

### What the backend does

- Loads the YOLOE weights from `config.json` (see `model.py`).
- Applies text prompts once at startup using `model.set_classes(prompts)`.
- Runs inference for each task image and outputs predictions in Label Studio `rectanglelabels` format.
- Applies post-inference Non-Max Suppression (NMS) to reduce overlapping detections.

### config.json parameters

- `model`: Local path for the weights file (relative paths are resolved under `my_ml_backend/`).
- `weights_url`: Optional download URL. If the weights file is missing, it will be downloaded to the `model` path.
- `imgsz`: Inference image size passed to Ultralytics.
- `conf`: Confidence threshold.
- `iou`: IoU threshold used for overlap filtering in the backend NMS step.
- `device`: Optional device override passed to Ultralytics (for example, `"cpu"`, `"0"`).
- `prompts`: List of text prompts (open-vocabulary classes) used by YOLOE.

### Utility functions

Helper functions are implemented in `utils.py`, including:

- Weights download if missing.
- NMS implementation driven by `config.json` `iou`.
- Conversion of model `xyxy` boxes into Label Studio percentage-based rectangles.

### Tests

`test_api.py` includes a mocked inference test that validates:

- Label Studio output formatting.
- NMS behavior changes based on the configured `iou` threshold.

### Dependencies

This backend requires `ultralytics` (see `requirements.txt`).


