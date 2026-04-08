"""Standalone training process for multiprocessing.

This module provides a picklable training function that runs in a separate
process, eliminating GIL contention with the main predict thread.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging for the child process
def _setup_logging() -> logging.Logger:
    """Setup logging for the training process."""
    logger = logging.getLogger("training_process")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _write_status(status_path: str, status: Dict[str, Any]) -> None:
    """Write training status to file."""
    Path(status_path).parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def train_in_process(
    job_id: str,
    dataset_summary: Dict[str, Any],
    config_dict: Dict[str, Any],
    runs_dir: str,
    datasets_dir: str,
    model_dir: str,
) -> None:
    """Run training in a separate process.
    
    This function must be picklable and accept all parameters as serializable
    arguments since it runs in a separate process via multiprocessing.
    
    Args:
        job_id: Unique training job identifier
        dataset_summary: Dataset build summary (root, data_yaml, etc.)
        config_dict: Serialized configuration dictionary
        runs_dir: Path to training runs directory
        datasets_dir: Path to datasets directory
        model_dir: Path to model state directory
    """
    logger = _setup_logging()
    
    # Set environment variables for ultralytics paths
    os.environ["TRAINING_RUNS_DIR"] = runs_dir
    os.environ["TRAINING_DATASETS_DIR"] = datasets_dir
    
    # Status file path
    run_dir = os.path.join(runs_dir, job_id)
    status_path = os.path.join(run_dir, "training_status.json")
    
    # Import ultralytics here to avoid pickling issues
    try:
        from ultralytics import YOLO, YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    except ImportError:
        YOLOE = None
        YOLOEPETrainer = None
    
    # Initial status
    start_time = time.time()
    _write_status(status_path, {
        "job_id": job_id,
        "status": "running",
        "epoch": 0,
        "total_epochs": config_dict.get("train_epochs", 20),
        "best_weights_path": None,
        "error": None,
        "started_at": start_time,
        "updated_at": start_time,
    })
    
    try:
        # Resolve training model path
        training_model_path = config_dict.get("train_init_weights") or config_dict.get("base_model")
        if not training_model_path:
            raise ValueError("No training model path configured")
        
        # Resolve to absolute path
        if not os.path.isabs(training_model_path):
            # Assume relative to model_dir
            training_model_path = os.path.join(model_dir, training_model_path)
        
        if not os.path.exists(training_model_path):
            raise FileNotFoundError(f"Training weights not found: {training_model_path}")
        
        # Training configuration
        data_yaml = dataset_summary.get("data_yaml")
        if not data_yaml or not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        
        imgsz = int(config_dict.get("imgsz", 640))
        device = config_dict.get("device")
        epochs = int(config_dict.get("train_epochs", 20))
        batch = config_dict.get("train_batch")
        batch = int(batch) if batch is not None else None
        freeze = config_dict.get("train_freeze")
        freeze = int(freeze) if freeze is not None else None
        
        # Build train kwargs
        train_kwargs = {
            "data": data_yaml,
            "imgsz": imgsz,
            "epochs": epochs,
            "project": runs_dir,
            "name": job_id,
        }
        if device is not None:
            train_kwargs["device"] = device
        if batch is not None:
            train_kwargs["batch"] = batch
        if freeze is not None:
            train_kwargs["freeze"] = freeze
        
        # Create YOLO model and train
        is_yoloe = isinstance(training_model_path, str) and "-seg" in training_model_path
        
        if is_yoloe:
            if YOLOE is None or YOLOEPETrainer is None:
                raise ValueError("YOLOE training requires YOLOE and YOLOEPETrainer")
            
            train_yoloe_yaml = config_dict.get("train_yoloe_yaml")
            if not train_yoloe_yaml:
                # Auto-resolve YAML
                yaml_path = _auto_resolve_yoloe_yaml(training_model_path, runs_dir)
                if not yaml_path:
                    raise ValueError("Cannot resolve YOLOE YAML for -seg checkpoint")
                train_yoloe_yaml = yaml_path
            
            yaml_path = _resolve_train_yoloe_yaml_path(train_yoloe_yaml, training_model_path)
            yolo = YOLOE(yaml_path, task="detect")
            yolo.load(training_model_path)
            train_kwargs["trainer"] = YOLOEPETrainer
            
            if bool(config_dict.get("train_linear_probe", False)):
                train_kwargs["freeze"] = _build_yoloe_linear_probe_freeze_list(yolo)
            
            logger.info(
                "Training job %s starting: epochs=%s imgsz=%s device=%s",
                job_id, epochs, imgsz, device,
            )
            
            # Run training with retry logic
            retried = False
            while True:
                try:
                    results = yolo.train(**train_kwargs)
                    break
                except RuntimeError as exc:
                    message = str(exc)
                    if (
                        not retried
                        and "PytorchStreamReader failed reading zip archive" in message
                        and _cleanup_ultralytics_mobileclip_assets()
                    ):
                        retried = True
                        logger.warning(
                            "YOLOE training failed due to corrupted MobileCLIP asset; retrying.",
                            exc_info=True,
                        )
                        continue
                    raise
        else:
            yolo = YOLO(training_model_path, task="detect")
            logger.info(
                "Training job %s starting: epochs=%s imgsz=%s device=%s",
                job_id, epochs, imgsz, device,
            )
            results = yolo.train(**train_kwargs)
        
        # Find best weights
        best_path = None
        best_pt = getattr(results, "best", None)
        if best_pt is not None:
            best_path = str(best_pt)
        
        save_dir = getattr(results, "save_dir", None)
        if save_dir is None:
            save_dir = getattr(getattr(yolo, "trainer", None), "save_dir", None)
        
        # Search for weights
        candidates = []
        if best_path:
            candidates.append(best_path)
        if isinstance(save_dir, (str, os.PathLike)):
            candidates.append(os.path.join(save_dir, "weights", "best.pt"))
            candidates.append(os.path.join(save_dir, "weights", "last.pt"))
        candidates.append(os.path.join(run_dir, "weights", "best.pt"))
        candidates.append(os.path.join(run_dir, "weights", "last.pt"))
        
        resolved_path = next((p for p in candidates if p and os.path.exists(p)), None)
        
        if not resolved_path:
            raise FileNotFoundError(
                f"Training did not produce weights. Checked: {candidates}"
            )
        
        # Write training metadata (class names from dataset) for use at approval time
        training_classes = dataset_summary.get("classes") or []
        training_meta = {
            "num_classes": len(training_classes),
            "class_names": list(training_classes),
        }
        training_meta_path = os.path.join(run_dir, "training_meta.json")
        try:
            _write_status(training_meta_path, training_meta)
            logger.info(
                "Training job %s wrote training_meta.json: %d classes",
                job_id,
                len(training_classes),
            )
        except OSError:
            logger.warning(
                "Training job %s failed to write training_meta.json",
                job_id,
                exc_info=True,
            )

        # Write success status
        _write_status(status_path, {
            "job_id": job_id,
            "status": "completed",
            "epoch": epochs,
            "total_epochs": epochs,
            "best_weights_path": resolved_path,
            "error": None,
            "started_at": start_time,
            "updated_at": time.time(),
            "save_dir": save_dir,
        })
        
        logger.info("Training job %s completed. Best weights: %s", job_id, resolved_path)
        
    except Exception as exc:
        logger.exception("Training job %s failed: %s", job_id, exc)
        _write_status(status_path, {
            "job_id": job_id,
            "status": "failed",
            "epoch": 0,
            "total_epochs": config_dict.get("train_epochs", 20),
            "best_weights_path": None,
            "error": str(exc),
            "started_at": start_time,
            "updated_at": time.time(),
        })
        raise


def _auto_resolve_yoloe_yaml(training_model_path: str, runs_dir: str) -> Optional[str]:
    """Auto-resolve YOLOE YAML path for a -seg checkpoint."""
    # Look for yaml directory relative to runs_dir or model path
    base_dir = os.path.dirname(os.path.dirname(training_model_path))
    yaml_dir = os.path.join(base_dir, "yaml")
    
    if not os.path.isdir(yaml_dir):
        return None
    
    # Determine scale from model filename
    stem = os.path.basename(training_model_path).lower()
    scale = None
    for s in ["n", "s", "m", "l", "x"]:
        if f"yoloe-{s}" in stem or f"yoloe_{s}" in stem:
            scale = s
            break
    
    if not scale:
        return None
    
    # Look for matching YAML files
    candidates = [
        f"yoloe-{scale}.yaml",
        f"yoloe_{scale}.yaml",
        f"yoloe-26{scale}.yaml",
        f"yoloe_26{scale}.yaml",
    ]
    
    for candidate in candidates:
        candidate_path = os.path.join(yaml_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    
    return None


def _resolve_train_yoloe_yaml_path(yaml_path: str, training_model_path: str) -> str:
    """Resolve YOLOE YAML path."""
    if os.path.isabs(yaml_path) and os.path.exists(yaml_path):
        return yaml_path
    
    # Try relative to model directory
    base_dir = os.path.dirname(os.path.dirname(training_model_path))
    candidate = os.path.join(base_dir, yaml_path)
    if os.path.exists(candidate):
        return candidate
    
    # Try as-is
    if os.path.exists(yaml_path):
        return yaml_path
    
    raise FileNotFoundError(f"YOLOE YAML not found: {yaml_path}")


def _build_yoloe_linear_probe_freeze_list(yolo) -> List[str]:
    """Build freeze list for YOLOE linear probe training."""
    freeze_list: List[str] = []
    try:
        model = getattr(yolo, "model", None)
        if model is None:
            return freeze_list
        
        for name, module in model.named_modules():
            if name.startswith("model."):
                name = name[6:]
            if name:
                freeze_list.append(name)
    except Exception:
        pass
    
    return freeze_list


def _cleanup_ultralytics_mobileclip_assets() -> bool:
    """Clean up corrupted MobileCLIP cache files."""
    try:
        import shutil
        from pathlib import Path
        
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        if not cache_dir.exists():
            return False
        
        removed = False
        for pattern in ["*mobileclip*", "*MobileCLIP*"]:
            for path in cache_dir.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        removed = True
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed = True
                except Exception:
                    pass
        
        return removed
    except Exception:
        return False
