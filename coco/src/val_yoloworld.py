from ultralytics import YOLOWorld
import argparse
import yaml
import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == '__main__':
    

    save_path = "yoloworld_pretrained"

    model = YOLOWorld("yolov8s-world")

    # Display model information (optional)
    model.info()

    # Validate the model
    metrics_test = model.val(data="coco.yaml",
                                save_json=True,
                                imgsz=640,
                                batch=100,
                                conf=0.25, iou=0.6, device="0", plots=True,
                                name=save_path + "/" + "test", 
                                exist_ok=True, split="val")

    metrics_categories = {
        "names": metrics_test.names,
        "map50-95": metrics_test.box.maps,
        "map50": metrics_test.box.ap50,
        "precision": metrics_test.box.p,
        "recall": metrics_test.box.r,
        "f1": metrics_test.box.f1
    }

    metrics = {
        "map50-95": metrics_test.box.map,
        "map50": metrics_test.box.map50,
        "map75": metrics_test.box.map75,
        "precision": metrics_test.box.mp,
        "recall": metrics_test.box.mr,
        "fitness": metrics_test.fitness,
        "categories": metrics_categories
    }

    json_obj = json.dumps(metrics, indent=4, cls=NumpyArrayEncoder)

    with open("runs/detect/" + save_path + "/" + "test" + "/test_metrics.json", "w") as outfile:
        outfile.write(json_obj)
