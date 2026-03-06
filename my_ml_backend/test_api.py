import json
import os

import pytest

from model import NewModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class _MockTensor(list):
    def tolist(self):
        return list(self)


class _MockBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _MockTensor(xyxy)
        self.conf = _MockTensor(conf)
        self.cls = _MockTensor(cls)

    def __len__(self):
        return len(self.conf)


class _MockResult:
    def __init__(self, boxes, orig_shape, names):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.names = names


class _MockYOLO:
    def __init__(self, *args, **kwargs):
        self.names = {0: "clapperboard"}
        self._classes = []

    def set_classes(self, classes):
        self._classes = classes

    def predict(self, **kwargs):
        boxes = _MockBoxes(
            xyxy=[
                [10.0, 10.0, 110.0, 110.0],
                [12.0, 12.0, 112.0, 112.0],
            ],
            conf=[0.9, 0.8],
            cls=[0, 0],
        )
        result = _MockResult(boxes=boxes, orig_shape=(200, 200), names=self.names)
        return [result]


@pytest.mark.parametrize(
    "configured_iou,expected_regions",
    [
        (0.99, 2),
        (0.50, 1),
    ],
)
def test_predict_nms_respects_config_iou(client, monkeypatch, configured_iou, expected_regions):
    import model as model_module

    monkeypatch.setattr(model_module, "YOLO", _MockYOLO)
    monkeypatch.setattr(model_module, "resolve_model_path", lambda *_args, **_kwargs: "mock.pt")
    monkeypatch.setattr(model_module.os.path, "exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        model_module,
        "load_backend_config",
        lambda *_args, **_kwargs: {
            "model": "weights/yoloe-26n-seg.pt",
            "imgsz": 640,
            "conf": 0.25,
            "iou": configured_iou,
            "device": None,
            "prompts": ["clapperboard"],
        },
    )

    request = {
        "tasks": [
            {
                "id": 1,
                "data": {
                    "image": "/tmp/fake.jpg",
                },
            }
        ],
        "project": "1.1",
        "label_config": """
        <View>
          <Image name=\"image\" value=\"$image\"/>
          <RectangleLabels name=\"label\" toName=\"image\">
            <Label value=\"clapperboard\"/>
          </RectangleLabels>
        </View>
        """,
    }

    response = client.post(
        "/predict",
        data=json.dumps(request),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = json.loads(response.data)

    assert "results" in payload
    assert len(payload["results"]) == 1
    assert "result" in payload["results"][0]

    regions = payload["results"][0]["result"]
    assert len(regions) == expected_regions

    for region in regions:
        assert region["type"] == "rectanglelabels"
        assert region["from_name"] == "label"
        assert region["to_name"] == "image"
        assert region["value"]["rectanglelabels"] == ["clapperboard"]

    region0 = regions[0]
    assert region0["value"]["x"] == 5.0
    assert region0["value"]["y"] == 5.0
    assert region0["value"]["width"] == 50.0
    assert region0["value"]["height"] == 50.0
