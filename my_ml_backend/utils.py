import json
import logging
import os
import shutil
import urllib.request
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = {
    "base_model": "weights/yoloe-26n-seg.pt",
    "weights_url": None,
    "train_init_weights": None,
    "train_weights_url": None,
    "train_yoloe_yaml": None,
    "train_dataset_prefix": None,
    "train_dataset_version": None,
    "train_dataset_per_job": False,
    "train_linear_probe": False,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.5,
    "device": None,
    "task": None,
    "mode": None,
    "prompt_mode": "batch",
    "prefer_image_fs": False,
    "prompts": [],
    "data_root": "/data",
    "train_epochs": 20,
    "train_batch": None,
    "train_freeze": None,
    "train_val_ratio": 0.1,
    "train_seed": 1337,
    "train_unify_label": None,
}


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportedBox:
    """Rectangle label parsed from an exported Label Studio annotation."""

    label: str
    x_center: float
    y_center: float
    width: float
    height: float


def iter_exported_annotation_files(annotations_root: str) -> Iterable[Path]:
    """Iterate over exported per-frame annotation JSON files."""

    root = Path(annotations_root)
    if not root.exists():
        return

    for file_id_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for json_path in sorted(file_id_dir.glob("*.json")):
            yield json_path


def parse_exported_annotation(annotation_path: Path) -> Tuple[str, str, List[ExportedBox]]:
    """Parse the exported annotation JSON into image path, file_id and boxes.

    Returns:
        Tuple[image_fs_path, file_id, boxes]
    """

    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data") or {}
    image_fs = data.get("image_fs")
    if not image_fs:
        raise ValueError(f"Missing data.image_fs in exported annotation: {annotation_path}")

    file_id = data.get("file_id")
    if not file_id:
        raise ValueError(f"Missing data.file_id in exported annotation: {annotation_path}")

    annotation = payload.get("annotation") or {}
    results = annotation.get("result") or []

    boxes: List[ExportedBox] = []
    for item in results:
        if item.get("type") != "rectanglelabels":
            continue

        value = item.get("value") or {}
        labels = value.get("rectanglelabels") or []
        if not labels:
            continue

        label = labels[0]
        x = float(value.get("x", 0.0))
        y = float(value.get("y", 0.0))
        w = float(value.get("width", 0.0))
        h = float(value.get("height", 0.0))

        x_center = (x + (w / 2.0)) / 100.0
        y_center = (y + (h / 2.0)) / 100.0
        width = w / 100.0
        height = h / 100.0

        boxes.append(
            ExportedBox(
                label=str(label),
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return str(image_fs), str(file_id), boxes


def load_or_create_splits(
    *,
    splits_path: Path,
    file_ids: Sequence[str],
    seed: int,
    val_ratio: float,
) -> Dict[str, List[str]]:
    """Load persisted train/val split or create one, preserving existing assignments."""

    if splits_path.exists():
        with splits_path.open("r", encoding="utf-8") as f:
            splits = json.load(f)
        train_ids = list(splits.get("train", []))
        val_ids = list(splits.get("val", []))
    else:
        train_ids, val_ids = [], []

    known = set(train_ids) | set(val_ids)
    new_ids = list(set(file_ids) - known)
    if new_ids:
        # Shuffle new IDs randomly
        random.shuffle(new_ids)
        val_count = max(1, int(len(new_ids) * val_ratio))
        val_ids.extend(new_ids[:val_count])
        train_ids.extend(new_ids[val_count:])

    splits = {"train": sorted(set(train_ids)), "val": sorted(set(val_ids))}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)

    return splits


def load_or_create_classes(classes_path: Path, labels: Sequence[str]) -> List[str]:
    """Load persisted class list or create a new one."""

    if classes_path.exists():
        with classes_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        classes = payload.get("classes")
        if isinstance(classes, list) and all(isinstance(item, str) for item in classes):
            return classes

    classes = sorted(set(str(label) for label in labels if label))
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    with classes_path.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2, sort_keys=True)

    return classes


def write_ultralytics_data_yaml(dataset_root: Path, classes: Sequence[str]) -> Path:
    """Write an Ultralytics-compatible data.yaml file."""

    yaml_path = dataset_root / "data.yaml"
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    for idx, name in enumerate(classes):
        escaped = str(name).replace("\"", "\\\"")
        lines.append(f"  {idx}: \"{escaped}\"")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def build_yolo_dataset_from_exports(
    *,
    data_root: str,
    dataset_root: str,
    seed: int,
    val_ratio: float,
    prompts: Optional[Sequence[str]] = None,
    unify_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a YOLO dataset from exported per-frame annotation JSON files.

    Args:
        data_root: Root directory containing `annotations/` and `frames/`.
        dataset_root: Output directory for the generated Ultralytics dataset.
        seed: Seed for deterministic split assignment.
        val_ratio: Ratio of file_ids assigned to validation.
        prompts: Optional class list override.
        unify_label: If set, all rectangle label names are mapped to this single canonical label.

    Returns:
        Summary information about the dataset build.
    """

    data_root_path = Path(data_root)
    annotations_root = data_root_path / "annotations"
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    annotation_files = list(iter_exported_annotation_files(str(annotations_root)))
    if not annotation_files:
        raise FileNotFoundError(f"No exported annotation JSON files found under {annotations_root}")

    parsed: List[Tuple[Path, str, str, List[ExportedBox]]] = []
    file_ids: List[str] = []
    discovered_labels: List[str] = []
    for path in annotation_files:
        image_fs, file_id, boxes = parse_exported_annotation(path)

        if unify_label:
            logger.info("Unifying exported labels to: %s", unify_label)
            boxes = [
                ExportedBox(
                    label=unify_label,
                    x_center=box.x_center,
                    y_center=box.y_center,
                    width=box.width,
                    height=box.height,
                )
                for box in boxes
            ]

        parsed.append((path, image_fs, file_id, boxes))
        file_ids.append(file_id)
        for box in boxes:
            discovered_labels.append(box.label)

    splits_path = dataset_root_path / "splits.json"
    splits = load_or_create_splits(
        splits_path=splits_path,
        file_ids=file_ids,
        seed=seed,
        val_ratio=val_ratio,
    )

    classes_path = dataset_root_path / "classes.json"
    if unify_label:
        classes = [unify_label]
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        with classes_path.open("w", encoding="utf-8") as f:
            json.dump({"classes": classes}, f, indent=2, sort_keys=True)
        logger.info("Wrote single-class classes.json due to unify_label: %s", classes)
    else:
        class_candidates = list(prompts) if prompts else discovered_labels
        classes = load_or_create_classes(classes_path, class_candidates)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    images_train_dir = dataset_root_path / "images" / "train"
    images_val_dir = dataset_root_path / "images" / "val"
    labels_train_dir = dataset_root_path / "labels" / "train"
    labels_val_dir = dataset_root_path / "labels" / "val"

    for p in (
        images_train_dir,
        images_val_dir,
        labels_train_dir,
        labels_val_dir,
    ):
        if p.exists():
            shutil.rmtree(p)
    for p in (images_train_dir, images_val_dir, labels_train_dir, labels_val_dir):
        p.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_boxes = 0
    skipped = 0

    for json_path, image_fs, file_id, boxes in parsed:
        split = "val" if file_id in set(splits["val"]) else "train"
        image_src = Path(image_fs)
        if not image_src.exists():
            logger.warning("Missing frame for annotation %s: %s", json_path, image_src)
            skipped += 1
            continue

        image_dst_dir = images_val_dir if split == "val" else images_train_dir
        label_dst_dir = labels_val_dir if split == "val" else labels_train_dir

        safe_file_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(file_id))
        output_stem = f"{safe_file_id}_{image_src.stem}"
        image_dst = image_dst_dir / f"{output_stem}{image_src.suffix}"
        if not image_dst.exists():
            shutil.copy2(image_src, image_dst)

        label_dst = label_dst_dir / f"{output_stem}.txt"
        label_lines: List[str] = []
        for box in boxes:
            if box.label not in class_to_id:
                continue
            class_id = class_to_id[box.label]
            label_lines.append(
                f"{class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
            )
            num_boxes += 1

        label_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        num_images += 1

    data_yaml_path = write_ultralytics_data_yaml(dataset_root_path, classes)

    return {
        "dataset_root": str(dataset_root_path),
        "data_yaml": str(data_yaml_path),
        "classes": classes,
        "num_images": num_images,
        "num_boxes": num_boxes,
        "skipped": skipped,
        "splits": splits,
    }


def load_backend_config(config_path: str) -> Dict:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("config.json must contain a JSON object")
        config.update(user_config)

    train_config_path = "/training/train_config.json"
    if os.path.exists(train_config_path):
        with open(train_config_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        if not isinstance(train_config, dict):
            raise ValueError("train_config.json must contain a JSON object")
        config.update(train_config)

    if config.get("weights_url") is not None and not isinstance(config["weights_url"], str):
        raise ValueError("weights_url must be a string or null")

    train_init_weights = config.get("train_init_weights")
    if train_init_weights is not None and not isinstance(train_init_weights, str):
        raise ValueError("train_init_weights must be a string or null")
    if isinstance(train_init_weights, str):
        train_init_weights = train_init_weights.strip()
        if not train_init_weights:
            train_init_weights = None
    config["train_init_weights"] = train_init_weights

    train_weights_url = config.get("train_weights_url")
    if train_weights_url is not None and not isinstance(train_weights_url, str):
        raise ValueError("train_weights_url must be a string or null")
    if isinstance(train_weights_url, str):
        train_weights_url = train_weights_url.strip()
        if not train_weights_url:
            train_weights_url = None
    config["train_weights_url"] = train_weights_url

    train_yoloe_yaml = config.get("train_yoloe_yaml")
    if train_yoloe_yaml is not None and not isinstance(train_yoloe_yaml, str):
        raise ValueError("train_yoloe_yaml must be a string or null")
    if isinstance(train_yoloe_yaml, str):
        train_yoloe_yaml = train_yoloe_yaml.strip()
        if not train_yoloe_yaml:
            train_yoloe_yaml = None
    config["train_yoloe_yaml"] = train_yoloe_yaml

    train_dataset_prefix = config.get("train_dataset_prefix")
    if train_dataset_prefix is not None and not isinstance(train_dataset_prefix, str):
        raise ValueError("train_dataset_prefix must be a string or null")
    if isinstance(train_dataset_prefix, str):
        train_dataset_prefix = train_dataset_prefix.strip()
        if not train_dataset_prefix:
            train_dataset_prefix = None
    config["train_dataset_prefix"] = train_dataset_prefix

    train_dataset_version = config.get("train_dataset_version")
    if train_dataset_version is not None and not isinstance(train_dataset_version, str):
        raise ValueError("train_dataset_version must be a string or null")
    if isinstance(train_dataset_version, str):
        train_dataset_version = train_dataset_version.strip()
        if not train_dataset_version:
            train_dataset_version = None
    config["train_dataset_version"] = train_dataset_version

    train_dataset_per_job = config.get("train_dataset_per_job", False)
    if not isinstance(train_dataset_per_job, bool):
        raise ValueError("train_dataset_per_job must be a boolean")
    config["train_dataset_per_job"] = train_dataset_per_job

    train_linear_probe = config.get("train_linear_probe", False)
    if not isinstance(train_linear_probe, bool):
        raise ValueError("train_linear_probe must be a boolean")
    config["train_linear_probe"] = train_linear_probe

    config["imgsz"] = int(config["imgsz"])
    if config["imgsz"] <= 0:
        raise ValueError("imgsz must be a positive integer")

    config["conf"] = float(config["conf"])
    if not 0.0 <= config["conf"] <= 1.0:
        raise ValueError("conf must be in range [0, 1]")

    config["iou"] = float(config["iou"])
    if not 0.0 <= config["iou"] <= 1.0:
        raise ValueError("iou must be in range [0, 1]")

    prompts = config.get("prompts") or []
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("prompts must be a list of strings")
    config["prompts"] = [item.strip() for item in prompts if item and item.strip()]

    data_root = config.get("data_root")
    if data_root is not None and not isinstance(data_root, str):
        raise ValueError("data_root must be a string")
    config["data_root"] = (data_root or "/data").strip()
    if not config["data_root"]:
        raise ValueError("data_root must not be empty")

    config["train_epochs"] = int(config.get("train_epochs", 20))
    if config["train_epochs"] <= 0:
        raise ValueError("train_epochs must be a positive integer")

    train_batch = config.get("train_batch")
    if train_batch is not None:
        train_batch = int(train_batch)
        if train_batch <= 0:
            raise ValueError("train_batch must be a positive integer")
    config["train_batch"] = train_batch

    train_freeze = config.get("train_freeze")
    if train_freeze is not None:
        train_freeze = int(train_freeze)
        if train_freeze < 0:
            raise ValueError("train_freeze must be a non-negative integer")
    config["train_freeze"] = train_freeze

    config["train_val_ratio"] = float(config.get("train_val_ratio", 0.1))
    if not 0.0 < config["train_val_ratio"] < 1.0:
        raise ValueError("train_val_ratio must be in range (0, 1)")

    config["train_seed"] = int(config.get("train_seed", 1337))

    train_unify_label = config.get("train_unify_label")
    if train_unify_label is not None and not isinstance(train_unify_label, str):
        raise ValueError("train_unify_label must be a string or null")
    if isinstance(train_unify_label, str):
        train_unify_label = train_unify_label.strip()
        if not train_unify_label:
            train_unify_label = None
    config["train_unify_label"] = train_unify_label

    if config.get("task") is not None and not isinstance(config["task"], str):
        raise ValueError("task must be a string or null")
    if config.get("mode") is not None and not isinstance(config["mode"], str):
        raise ValueError("mode must be a string or null")

    if config.get("prompt_mode") is None:
        config["prompt_mode"] = "batch"
    if not isinstance(config["prompt_mode"], str):
        raise ValueError("prompt_mode must be a string")
    if config["prompt_mode"] not in {"batch", "per_prompt"}:
        raise ValueError("prompt_mode must be one of: batch, per_prompt")

    if not isinstance(config.get("prefer_image_fs"), bool):
        raise ValueError("prefer_image_fs must be a boolean")

    return config


def resolve_model_path(model_path: str, base_dir: str) -> str:
    if not model_path:
        raise ValueError("Model path is empty")

    expanded = os.path.expanduser(model_path)
    if os.path.isabs(expanded):
        resolved = expanded
    else:
        resolved = os.path.join(base_dir, expanded)

    resolved = os.path.abspath(resolved)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Model weights file not found: {resolved}")

    return resolved


def download_weights_if_needed(
    model_path: str,
    base_dir: str,
    weights_url: str | None,
) -> str:
    expanded = os.path.expanduser(model_path)
    target_path = expanded if os.path.isabs(expanded) else os.path.join(base_dir, expanded)
    target_path = os.path.abspath(target_path)

    if os.path.exists(target_path):
        return target_path

    if not weights_url:
        raise FileNotFoundError(
            f"Model weights file not found: {target_path}. "
            "Provide 'weights_url' in config.json to download it automatically."
        )

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    tmp_path = f"{target_path}.tmp"
    try:
        with urllib.request.urlopen(weights_url) as response, open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return target_path


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float,
    class_agnostic: bool = False,
) -> List[Dict]:
    if not detections:
        return []

    pending = sorted(detections, key=lambda det: det.get("score", 0.0), reverse=True)
    kept = []

    while pending:
        current = pending.pop(0)
        kept.append(current)

        survivors = []
        for candidate in pending:
            if not class_agnostic and candidate.get("class_id") != current.get("class_id"):
                survivors.append(candidate)
                continue

            overlap = box_iou(current["xyxy"], candidate["xyxy"])
            if overlap <= iou_threshold:
                survivors.append(candidate)

        pending = survivors

    return kept


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def xyxy_to_ls_percent(xyxy: List[float], image_width: int, image_height: int) -> Dict[str, float]:
    x1, y1, x2, y2 = xyxy

    x1 = _clamp(float(x1), 0.0, float(image_width))
    y1 = _clamp(float(y1), 0.0, float(image_height))
    x2 = _clamp(float(x2), 0.0, float(image_width))
    y2 = _clamp(float(y2), 0.0, float(image_height))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return {
        "x": (x1 / image_width) * 100.0,
        "y": (y1 / image_height) * 100.0,
        "width": (width / image_width) * 100.0,
        "height": (height / image_height) * 100.0,
    }
