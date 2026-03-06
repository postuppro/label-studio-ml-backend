import json
import os
import shutil
import urllib.request
from typing import Dict, List


DEFAULT_CONFIG = {
    "model": "weights/yoloe-26n-seg.pt",
    "weights_url": None,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.5,
    "device": None,
    "prompts": [],
}


def load_backend_config(config_path: str) -> Dict:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("config.json must contain a JSON object")
        config.update(user_config)

    if config.get("weights_url") is not None and not isinstance(config["weights_url"], str):
        raise ValueError("weights_url must be a string or null")

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
