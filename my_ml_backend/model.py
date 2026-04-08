import logging
import multiprocessing
import os
import inspect
import json
import signal
import threading
import time
import uuid
from typing import List, Dict, Optional

from ultralytics import YOLO
import ultralytics

try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer
except Exception:
    YOLOE = None
    YOLOEPETrainer = None
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from utils import (
    build_yolo_dataset_from_exports,
    download_weights_if_needed,
    load_backend_config,
    non_max_suppression,
    resolve_model_path,
    xyxy_to_ls_percent,
)

from training_process import train_in_process

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.base_dir, "config.json")
        self.config = load_backend_config(config_path)

        self._training_lock = threading.Lock()
        self._training_process: Optional[multiprocessing.Process] = None

        download_weights_if_needed(
            model_path=self.config["base_model"],
            base_dir=self.base_dir,
            weights_url=self.config.get("weights_url"),
        )

        self._load_model_state()

        approved_weights_path = self.get("approved_weights_path")
        using_approved_model = (
            approved_weights_path
            and isinstance(approved_weights_path, str)
            and os.path.exists(approved_weights_path)
        )
        if using_approved_model:
            model_path = approved_weights_path
            logger.info("Loading approved weights for project %s: %s", self.project_id, model_path)
        else:
            model_path = resolve_model_path(self.config["base_model"], self.base_dir)
        constructor_kwargs = {}
        try:
            allowed_params = set(inspect.signature(YOLO).parameters)
        except (TypeError, ValueError):
            allowed_params = set()

        if self.config.get("task") is not None and "task" in allowed_params:
            constructor_kwargs["task"] = self.config["task"]
        if self.config.get("mode") is not None and "mode" in allowed_params:
            constructor_kwargs["mode"] = self.config["mode"]

        if using_approved_model:
            # Approved custom-trained models are loaded as plain YOLO
            self.model = self._create_inference_model(
                model_path,
                constructor_kwargs,
                use_plain_yolo=True,
            )
        else:
            # Determine YAML path for YOLOE end2end support
            yaml_path = None
            base_yaml = self.config.get("base_model_yaml")
            if base_yaml:
                try:
                    yaml_path = self._resolve_train_yoloe_yaml_path(
                        str(base_yaml), model_path
                    )
                except FileNotFoundError:
                    yaml_path = None

            self.model = self._create_inference_model(
                model_path,
                constructor_kwargs,
                yaml_path=yaml_path,
            )
        self._loaded_weights_path = model_path

        # Only apply YOLOE prompts for the base model, not approved custom models
        if not using_approved_model:
            prompts = self.config.get("prompts", [])
            if prompts:
                try:
                    self._try_set_model_classes(self.model, prompts)
                except RuntimeError as exc:
                    message = str(exc)
                    if "PytorchStreamReader failed reading zip archive" not in message:
                        raise

                    candidate_paths = [
                        "/tmp/Ultralytics/mobileclip2_b.ts",
                        os.path.expanduser("~/.config/Ultralytics/mobileclip2_b.ts"),
                        os.path.expanduser("~/.cache/Ultralytics/mobileclip2_b.ts"),
                    ]
                    removed_any = False
                    for candidate in candidate_paths:
                        try:
                            if os.path.exists(candidate):
                                os.remove(candidate)
                                removed_any = True
                        except OSError:
                            continue

                    if removed_any:
                        try:
                            self._try_set_model_classes(self.model, prompts)
                        except Exception:
                            logger.warning(
                                "Prompt embedding initialization failed after asset cleanup; continuing without prompts.",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            "Prompt embedding initialization failed; continuing without prompts.",
                            exc_info=True,
                        )

        self.set("model_version", os.path.basename(model_path))

    def get_active_model_info(self) -> Dict[str, Optional[str]]:
        """Return the currently active (approved) model info.

        Returns:
            Dict with keys: approved_weights_path, approved_job_id, approved_at,
            approved_yaml_path, approved_class_names.
        """

        approved_weights_path = self.get("approved_weights_path")
        approved_job_id = self.get("approved_job_id")
        approved_at = self.get("approved_at")
        approved_yaml_path = self.get("approved_yaml_path")
        approved_class_names = self._get_approved_class_names()
        return {
            "approved_weights_path": str(approved_weights_path) if approved_weights_path else None,
            "approved_job_id": str(approved_job_id) if approved_job_id else None,
            "approved_at": str(approved_at) if approved_at else None,
            "approved_yaml_path": str(approved_yaml_path) if approved_yaml_path else None,
            "approved_class_names": approved_class_names,
        }

    def approve_model(self, job_id: str) -> Dict[str, Optional[str]]:
        """Set a trained run as the active (approved) model.

        Resolves the weights path from the job ID using the training runs
        directory convention: ``<TRAINING_RUNS_DIR>/<job_id>/weights/best.pt``.
        The model is immediately loaded for inference. State is persisted to
        disk so it survives container restarts.

        Args:
            job_id: The training run / job identifier (directory name under runs/).

        Returns:
            Updated active model info.
        """

        if not job_id:
            raise ValueError("job_id is required")

        weights_path = os.path.join(self._get_runs_dir(), str(job_id), "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found for job {job_id}: {weights_path}"
            )

        self.set("approved_weights_path", weights_path)
        self.set("approved_job_id", job_id)
        self.set("approved_at", str(time.time()))
        self.set("approved_yaml_path", "")

        # Read and persist training class names for the approved model
        class_names = self._read_training_class_names(job_id)
        if class_names:
            self.set("approved_class_names", json.dumps(class_names))
        else:
            self.set("approved_class_names", "")
            logger.warning(
                "No training class metadata found for job %s; "
                "class names will be resolved from model.names at prediction time.",
                job_id,
            )

        self.set("model_version", os.path.basename(weights_path))
        self._save_model_state()

        # Load approved model as plain YOLO (not YOLOE) — no set_classes needed
        with self._training_lock:
            constructor_kwargs = {}
            try:
                allowed_params = set(inspect.signature(YOLO).parameters)
            except (TypeError, ValueError):
                allowed_params = set()
            if self.config.get("task") is not None and "task" in allowed_params:
                constructor_kwargs["task"] = self.config["task"]
            if self.config.get("mode") is not None and "mode" in allowed_params:
                constructor_kwargs["mode"] = self.config["mode"]

            self.model = self._create_inference_model(
                weights_path,
                constructor_kwargs,
                use_plain_yolo=True,
            )
            self._loaded_weights_path = weights_path

        return self.get_active_model_info()

    def reset_active_model(self) -> Dict[str, Optional[str]]:
        """Clear the approved model and revert to the base model.

        Returns:
            Active model info (all None after reset).
        """

        self.set("approved_weights_path", "")
        self.set("approved_job_id", "")
        self.set("approved_at", "")
        self.set("approved_yaml_path", "")
        self.set("approved_class_names", "")
        self._save_model_state()

        base_model_path = resolve_model_path(self.config["base_model"], self.base_dir)
        with self._training_lock:
            constructor_kwargs = {}
            try:
                allowed_params = set(inspect.signature(YOLO).parameters)
            except (TypeError, ValueError):
                allowed_params = set()
            if self.config.get("task") is not None and "task" in allowed_params:
                constructor_kwargs["task"] = self.config["task"]
            if self.config.get("mode") is not None and "mode" in allowed_params:
                constructor_kwargs["mode"] = self.config["mode"]

            # Use base_model_yaml if available (for custom base model architecture)
            base_yaml = self.config.get("base_model_yaml")
            yaml_path = None
            if base_yaml:
                try:
                    yaml_path = self._resolve_train_yoloe_yaml_path(
                        str(base_yaml), base_model_path
                    )
                except FileNotFoundError:
                    yaml_path = None

            self.model = self._create_inference_model(
                base_model_path,
                constructor_kwargs,
                yaml_path=yaml_path,
            )
            self._loaded_weights_path = base_model_path

            prompts = self.config.get("prompts", [])
            if prompts:
                self._try_set_model_classes(self.model, prompts)

        self.set("model_version", os.path.basename(base_model_path))
        return self.get_active_model_info()

    def _create_inference_model(
        self,
        model_path: str,
        constructor_kwargs: Dict[str, object],
        yaml_path: Optional[str] = None,
        use_plain_yolo: bool = False,
    ):
        """Create an inference model instance.

        When ``use_plain_yolo`` is True the model is loaded as a standard
        ``YOLO`` instance — used for approved custom-trained models whose
        class names are baked into the ``.pt`` weights.

        Otherwise, always uses the ``YOLOE`` wrapper when available.
        ``YOLOE`` can load any YOLO checkpoint and is required for
        prompt-based inference via ``set_classes()``.  Falls back to plain
        ``YOLO`` only when the ``YOLOE`` class was not importable.

        When ``yaml_path`` is provided and ``YOLOE`` is available, the model
        architecture is loaded from the YAML first, then weights are loaded
        from ``model_path``. This is necessary for ``end2end`` models to ensure
        correct tensor shapes during inference.

        Args:
            model_path: Path to the model weights file.
            constructor_kwargs: Additional keyword arguments for the YOLO constructor.
            yaml_path: Optional path to YOLOE YAML config for architecture-first loading.
            use_plain_yolo: If True, always load as plain ``YOLO`` (no YOLOE/yaml).

        Returns:
            A YOLOE or YOLO model instance.
        """

        if use_plain_yolo:
            logger.info("Loading approved custom model as plain YOLO: %s", model_path)
            return YOLO(model_path, **constructor_kwargs)

        if YOLOE is not None:
            if yaml_path:
                model = YOLOE(yaml_path, task="detect")
                model.load(model_path)
                return model
            return YOLOE(model_path)
        return YOLO(model_path, **constructor_kwargs)

    @staticmethod
    def _try_set_model_classes(model: object, class_names: List[str]) -> bool:
        """Set class prompts on the model via ``set_classes()``.

        Since we always load with the ``YOLOE`` wrapper (when available),
        ``set_classes`` computes text embeddings internally.

        Returns:
            True if classes were set, False if ``set_classes`` is unavailable.
        """

        if not class_names:
            return False

        set_classes = getattr(model, "set_classes", None)
        if set_classes is None:
            return False

        set_classes(class_names)
        return True

    @staticmethod
    def _resolve_training_model_path(base_model_path: str) -> str:
        """Resolve a detection-compatible base model path for training."""

        if not isinstance(base_model_path, str):
            return base_model_path

        if "-seg" in base_model_path:
            candidate = base_model_path.replace("-seg", "")
            if os.path.exists(candidate):
                return candidate

        return base_model_path

    @staticmethod
    def _auto_resolve_yoloe_yaml(training_model_path: str) -> Optional[str]:
        """Try to locate the YOLOE YAML inside the installed ultralytics package.

        Ultralytics recommends initializing a YOLOE detection model from a YAML config
        (e.g. yoloe-26n.yaml) and then loading a pretrained segmentation checkpoint
        (e.g. yoloe-26n-seg.pt).
        """

        if not isinstance(training_model_path, str) or not training_model_path:
            return None

        basename = os.path.basename(training_model_path)
        if not basename.endswith("-seg.pt"):
            return None

        yaml_name = basename.replace("-seg.pt", ".yaml")
        try:
            pkg_root = os.path.dirname(os.path.abspath(ultralytics.__file__))
        except Exception:
            return None

        for root, _, files in os.walk(pkg_root):
            if yaml_name in files:
                return os.path.join(root, yaml_name)

        return None

    @staticmethod
    def _build_yoloe_linear_probe_freeze_list(yoloe_model: object) -> List[str]:
        if not hasattr(yoloe_model, "model"):
            raise ValueError("Unexpected YOLOE model structure: missing model")

        inner = getattr(yoloe_model.model, "model", None)
        if inner is None or not hasattr(inner, "__len__"):
            raise ValueError("Unexpected YOLOE model structure: missing model.model")

        head_index = len(inner) - 1
        freeze: List[str] = [str(i) for i in range(0, head_index)]

        head = inner[head_index]
        if hasattr(head, "named_children"):
            for name, _child in head.named_children():
                if "cv3" not in name:
                    freeze.append(f"{head_index}.{name}")

        freeze.extend(
            [
                f"{head_index}.cv3.0.0",
                f"{head_index}.cv3.0.1",
                f"{head_index}.cv3.1.0",
                f"{head_index}.cv3.1.1",
                f"{head_index}.cv3.2.0",
                f"{head_index}.cv3.2.1",
            ]
        )
        return freeze

    def _resolve_train_yoloe_yaml_path(self, train_yoloe_yaml: str, training_model_path: str) -> str:
        if not train_yoloe_yaml:
            raise ValueError("train_yoloe_yaml is empty")

        yaml_candidate = str(train_yoloe_yaml)

        try:
            yaml_path = resolve_model_path(yaml_candidate, self.base_dir)
            return yaml_path
        except FileNotFoundError:
            pass

        auto_yaml = self._auto_resolve_yoloe_yaml(str(training_model_path))
        if auto_yaml:
            return auto_yaml

        raise FileNotFoundError(
            f"YOLOE model YAML not found: '{train_yoloe_yaml}'. "
            "Provide an absolute path or a path relative to /app."
        )

    def _write_training_run_snapshots(self, run_dir: str, dataset_summary: Dict[str, object]) -> None:
        if not run_dir:
            return

        try:
            os.makedirs(run_dir, exist_ok=True)
            train_config_snapshot = {
                key: value
                for key, value in self.config.items()
                if key == "data_root" or str(key).startswith("train_")
            }
            with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
                json.dump(train_config_snapshot, f, indent=2, sort_keys=True)
            with open(os.path.join(run_dir, "ml_backend_train_config.json"), "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            with open(os.path.join(run_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
                json.dump(dataset_summary, f, indent=2, sort_keys=True, default=str)
        except OSError:
            logger.warning("Failed to write training snapshots to run directory: %s", run_dir, exc_info=True)

    @staticmethod
    def _cleanup_ultralytics_mobileclip_assets() -> bool:
        candidate_paths = [
            "/tmp/Ultralytics/mobileclip2_b.ts",
            os.path.expanduser("~/.config/Ultralytics/mobileclip2_b.ts"),
            os.path.expanduser("~/.cache/Ultralytics/mobileclip2_b.ts"),
        ]

        removed_any = False
        for candidate in candidate_paths:
            try:
                if os.path.exists(candidate):
                    os.remove(candidate)
                    removed_any = True
            except OSError:
                continue

        return removed_any

    def _get_model_dir(self) -> str:
        """Return the absolute path to the model directory."""
        model_dir = os.getenv("MODEL_DIR") or os.path.join(self.base_dir, "data")
        return os.path.abspath(model_dir)

    def _get_runs_dir(self) -> str:
        """Return the absolute path to the training runs directory.

        Uses ``TRAINING_RUNS_DIR`` env var if set, otherwise falls back to
        ``<MODEL_DIR>/runs``.
        """
        runs_dir = os.getenv("TRAINING_RUNS_DIR")
        if runs_dir:
            return os.path.abspath(runs_dir)
        return os.path.join(self._get_model_dir(), "runs")

    def _get_datasets_dir(self) -> str:
        """Return the absolute path to the training datasets directory.

        Uses ``TRAINING_DATASETS_DIR`` env var if set, otherwise falls back to
        ``<MODEL_DIR>/datasets``.
        """
        datasets_dir = os.getenv("TRAINING_DATASETS_DIR")
        if datasets_dir:
            return os.path.abspath(datasets_dir)
        return os.path.join(self._get_model_dir(), "datasets")

    _STATE_KEYS = (
        "approved_weights_path",
        "approved_job_id",
        "approved_at",
        "approved_yaml_path",
        "approved_class_names",
    )

    def _get_state_file_path(self) -> str:
        """Return the path to the persistent model state file."""
        return os.path.join(self._get_model_dir(), "model_state.json")

    def _load_model_state(self) -> None:
        """Load persisted model state from disk into the in-memory cache.

        Called during ``setup()`` so that approval and training state survives
        container restarts.
        """
        state_path = self._get_state_file_path()
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return
            for key in self._STATE_KEYS:
                value = state.get(key)
                if value is not None:
                    self.set(key, value)
            logger.info("Loaded model state from %s", state_path)
        except Exception:
            logger.warning("Failed to load model state from %s", state_path, exc_info=True)

    def _save_model_state(self) -> None:
        """Persist current model state to disk.

        Writes the relevant cache keys to a JSON file in the model directory
        so they survive container restarts.
        """
        state = {}
        for key in self._STATE_KEYS:
            value = self.get(key)
            if value is not None:
                state[key] = str(value)
        state_path = self._get_state_file_path()
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
        except Exception:
            logger.warning("Failed to save model state to %s", state_path, exc_info=True)

    def _get_approved_class_names(self) -> Optional[List[str]]:
        """Return the persisted approved class names, or None if absent.

        The value is stored as a JSON-encoded string in the in-memory cache
        and in the model state file.
        """

        raw = self.get("approved_class_names")
        if not raw or not isinstance(raw, str):
            return None
        try:
            names = json.loads(raw)
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return names
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _read_training_class_names(self, job_id: str) -> Optional[List[str]]:
        """Read class names from the training run directory.

        Checks ``training_meta.json`` first, then falls back to
        ``dataset_summary.json``.

        Args:
            job_id: Training run / job identifier.

        Returns:
            List of class name strings, or None if not found.
        """

        run_dir = os.path.join(self._get_runs_dir(), str(job_id))
        for filename in ("training_meta.json", "dataset_summary.json"):
            meta_path = os.path.join(run_dir, filename)
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                names = meta.get("class_names") or meta.get("classes")
                if isinstance(names, list) and all(isinstance(n, str) for n in names):
                    logger.info(
                        "Read %d class names from %s for job %s",
                        len(names), filename, job_id,
                    )
                    return names
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Failed to read class names from %s for job %s",
                    meta_path, job_id, exc_info=True,
                )
        return None

    @staticmethod
    def _resolve_data_key(value: str) -> str:
        if isinstance(value, str) and value.startswith("$"):
            return value[1:]
        return value

    def _build_regions(self, result, from_name: str, to_name: str, label_map: Dict[str, str]) -> List[Dict]:
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        names = result.names if hasattr(result, "names") else self.model.names
        detections = []

        for xyxy, score, cls_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            class_id = int(cls_id)
            model_label = names[class_id]

            if label_map:
                if model_label not in label_map:
                    continue
                output_label = label_map[model_label]
            else:
                output_label = model_label

            detections.append(
                {
                    "xyxy": [float(v) for v in xyxy],
                    "score": float(score),
                    "class_id": class_id,
                    "label": output_label,
                }
            )

        filtered_detections = non_max_suppression(
            detections,
            iou_threshold=float(self.config["iou"]),
            class_agnostic=True,
        )

        image_height, image_width = result.orig_shape
        regions = []
        for det in filtered_detections:
            box_percent = xyxy_to_ls_percent(det["xyxy"], image_width, image_height)
            regions.append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [det["label"]],
                        **box_percent,
                    },
                    "score": det["score"],
                }
            )

        return regions

    def _detections_to_regions(
        self,
        detections: List[Dict],
        orig_shape,
        from_name: str,
        to_name: str,
    ) -> List[Dict]:
        if not detections:
            return []

        image_height, image_width = orig_shape
        regions = []
        for det in detections:
            box_percent = xyxy_to_ls_percent(det["xyxy"], image_width, image_height)
            regions.append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [det["label"]],
                        **box_percent,
                    },
                    "score": det["score"],
                }
            )

        return regions

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Write your inference logic here.

        Args:
            tasks: Label Studio tasks in JSON format.
            context: Label Studio context in JSON format.

        Returns:
            Model response.
        """

        # Determine if an approved custom model is active
        approved_weights_path = self.get("approved_weights_path")
        using_approved_model = (
            approved_weights_path
            and isinstance(approved_weights_path, str)
            and os.path.exists(approved_weights_path)
        )

        # Hot-reload approved model if weights changed since last predict
        if (
            using_approved_model
            and getattr(self, "_loaded_weights_path", None) != approved_weights_path
        ):
            with self._training_lock:
                if getattr(self, "_loaded_weights_path", None) != approved_weights_path:
                    constructor_kwargs = {}
                    try:
                        allowed_params = set(inspect.signature(YOLO).parameters)
                    except (TypeError, ValueError):
                        allowed_params = set()
                    if self.config.get("task") is not None and "task" in allowed_params:
                        constructor_kwargs["task"] = self.config["task"]
                    if self.config.get("mode") is not None and "mode" in allowed_params:
                        constructor_kwargs["mode"] = self.config["mode"]
                    # Approved custom models load as plain YOLO
                    self.model = self._create_inference_model(
                        approved_weights_path,
                        constructor_kwargs,
                        use_plain_yolo=True,
                    )
                    self._loaded_weights_path = approved_weights_path

        from_name, to_name, value = self.get_first_tag_occurence("RectangleLabels", "Image")
        data_key = self._resolve_data_key(value)

        if using_approved_model:
            # Custom model path: class names from training metadata or model.names
            approved_class_names = self._get_approved_class_names()
            if approved_class_names:
                model_names = approved_class_names
            else:
                model_names = list(self.model.names.values())
            label_map = self.build_label_map(from_name, model_names)
            # Disable prompt-based logic for custom models
            prompts = []
            prompt_mode = "batch"
        else:
            # Base model path: YOLOE prompts (unchanged)
            prompt_mode = self.config.get("prompt_mode", "batch")
            prompts = self.config.get("prompts", [])

            if prompts and hasattr(self.model, "set_classes"):
                self._try_set_model_classes(self.model, prompts)

            if prompts:
                model_names = list(prompts)
            else:
                model_names = list(self.model.names.values())

            label_map = self.build_label_map(from_name, model_names)

        predictions = []
        for task in tasks:
            task_data = task.get("data", {})
            if self.config.get("prefer_image_fs") and task_data.get("image_fs"):
                source = task_data.get("image_fs")
                logger.debug(
                    "Task %s using image_fs source: %s",
                    task.get("id"),
                    source,
                )
            else:
                source = task_data.get(data_key) or task_data.get(value)
                logger.debug(
                    "Task %s using fallback image source (key=%s): %s",
                    task.get("id"),
                    data_key,
                    source,
                )
            if not source:
                logger.warning(
                    "Task %s has no image source for key '%s'",
                    task.get("id"),
                    data_key,
                )
                predictions.append(
                    {
                        "result": [],
                        "score": 0.0,
                        "model_version": str(self.model_version),
                    }
                )
                continue

            image_path = (
                source
                if os.path.exists(source)
                else self.get_local_path(
                    source,
                    ls_host=os.getenv("LABEL_STUDIO_URL"),
                    task_id=task.get("id"),
                )
            )

            predict_kwargs = {
                "source": image_path,
                "conf": float(self.config["conf"]),
                "iou": float(self.config["iou"]),
                "imgsz": int(self.config["imgsz"]),
                "verbose": False,
            }

            if self.config.get("task") is not None:
                predict_kwargs["task"] = self.config["task"]
            if self.config.get("mode") is not None:
                predict_kwargs["mode"] = self.config["mode"]

            if self.config.get("device") is not None:
                predict_kwargs["device"] = self.config["device"]

            allowed_params = set(inspect.signature(self.model.predict).parameters)
            filtered_predict_kwargs = {
                k: v for k, v in predict_kwargs.items() if k in allowed_params
            }

            if prompts and prompt_mode == "per_prompt":
                merged_detections: List[Dict] = []
                orig_shape = None

                for prompt in prompts:
                    self._try_set_model_classes(self.model, [prompt])
                    results = self.model.predict(**filtered_predict_kwargs)
                    if not results:
                        continue
                    result0 = results[0]
                    if orig_shape is None and hasattr(result0, "orig_shape"):
                        orig_shape = result0.orig_shape

                    boxes = getattr(result0, "boxes", None)
                    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                        continue

                    for xyxy, score, cls_id in zip(
                        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                    ):
                        model_label = prompt
                        if label_map:
                            if model_label not in label_map:
                                continue
                            output_label = label_map[model_label]
                        else:
                            output_label = model_label

                        merged_detections.append(
                            {
                                "xyxy": [float(v) for v in xyxy],
                                "score": float(score),
                                "class_id": int(cls_id),
                                "label": output_label,
                            }
                        )

                # Restore full prompt set after per-prompt loop
                self._try_set_model_classes(self.model, prompts)

                filtered_detections = non_max_suppression(
                    merged_detections,
                    iou_threshold=float(self.config["iou"]),
                    class_agnostic=True,
                )
                regions = (
                    self._detections_to_regions(filtered_detections, orig_shape, from_name, to_name)
                    if orig_shape is not None
                    else []
                )
            else:
                # Batch mode (classes already set before loop) or no prompts
                results = self.model.predict(**filtered_predict_kwargs)
                regions = self._build_regions(results[0], from_name, to_name, label_map)
            avg_score = sum(region["score"] for region in regions) / max(len(regions), 1)

            predictions.append(
                {
                    "result": regions,
                    "score": avg_score,
                    "model_version": str(self.model_version),
                }
            )

        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        if event != "START_TRAINING":
            self.set("needs_retrain", "true")
            return {"status": "ignored", "event": event}

        status = self.get("training_job_status")
        if status == "running":
            active_process = self._training_process
            if active_process is None or not active_process.is_alive():
                logger.warning(
                    "Clearing stale training_job_status=running for project %s (no active training process)",
                    self.project_id,
                )
                self.set("training_job_status", "failed")
                self.set("training_job_error", "Stale running state cleared after restart")
                self.set("training_job_completed_at", str(time.time()))
            else:
                return {
                    "status": "already_running",
                    "job_id": self.get("training_job_id"),
                }

        job_id = uuid.uuid4().hex
        self.set("training_job_id", job_id)
        self.set("training_job_status", "queued")
        self.set("training_job_error", "")

        data_root = str(self.config.get("data_root", "/data"))
        dataset_prefix = self.config.get("train_dataset_prefix")
        dataset_version = self.config.get("train_dataset_version")

        if dataset_prefix and dataset_version:
            safe_prefix = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(dataset_prefix)
            ).strip("_-")
            safe_version = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(dataset_version)
            ).strip("_-")
            dataset_dirname = f"{safe_prefix}_{safe_version}" if safe_prefix and safe_version else f"project_{self.project_id}"
        else:
            dataset_dirname = f"project_{self.project_id}"

        dataset_root = os.path.join(self._get_datasets_dir(), dataset_dirname)

        val_ratio = float(self.config.get("train_val_ratio", 0.1))
        seed = int(self.config.get("train_seed", 1337))
        prompts = self.config.get("prompts") or None
        unify_label = self.config.get("train_unify_label")

        dataset_summary = build_yolo_dataset_from_exports(
            data_root=data_root,
            dataset_root=dataset_root,
            seed=seed,
            val_ratio=val_ratio,
            prompts=prompts,
            unify_label=unify_label,
        )

        self.set("needs_retrain", "false")
        self.set("training_dataset_root", str(dataset_summary.get("dataset_root")))
        self.set("training_data_yaml", str(dataset_summary.get("data_yaml")))

        # Prepare config dict for serialization to child process
        config_dict = dict(self.config)
        
        training_process = multiprocessing.Process(
            target=train_in_process,
            args=(
                job_id,
                dataset_summary,
                config_dict,
                self._get_runs_dir(),
                self._get_datasets_dir(),
                self._get_model_dir(),
            ),
            daemon=True,
        )
        self._training_process = training_process
        training_process.start()

        return {
            "status": "queued",
            "job_id": job_id,
            "dataset": {
                "root": dataset_summary.get("dataset_root"),
                "num_images": dataset_summary.get("num_images"),
                "num_boxes": dataset_summary.get("num_boxes"),
                "skipped": dataset_summary.get("skipped"),
            },
        }

