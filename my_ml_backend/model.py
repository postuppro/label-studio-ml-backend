import logging
import os
from typing import List, Dict, Optional

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from utils import (
    download_weights_if_needed,
    load_backend_config,
    non_max_suppression,
    resolve_model_path,
    xyxy_to_ls_percent,
)


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

        download_weights_if_needed(
            model_path=self.config["model"],
            base_dir=self.base_dir,
            weights_url=self.config.get("weights_url"),
        )
        model_path = resolve_model_path(self.config["model"], self.base_dir)
        self.model = YOLO(model_path)

        prompts = self.config.get("prompts", [])
        if prompts:
            self.model.set_classes(prompts)

        self.set("model_version", os.path.basename(model_path))

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

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            "Run prediction on %d tasks, project ID=%s",
            len(tasks),
            self.project_id,
        )

        from_name, to_name, value = self.get_first_tag_occurence("RectangleLabels", "Image")
        data_key = self._resolve_data_key(value)

        model_names = list(self.model.names.values())
        label_map = self.build_label_map(from_name, model_names)

        predictions = []
        for task in tasks:
            task_data = task.get("data", {})
            source = task_data.get(data_key) or task_data.get(value)
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
                else self.get_local_path(source, task_id=task.get("id"))
            )

            predict_kwargs = {
                "source": image_path,
                "conf": float(self.config["conf"]),
                "iou": float(self.config["iou"]),
                "imgsz": int(self.config["imgsz"]),
                "verbose": False,
            }
            if self.config.get("device") is not None:
                predict_kwargs["device"] = self.config["device"]

            results = self.model.predict(**predict_kwargs)
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

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

