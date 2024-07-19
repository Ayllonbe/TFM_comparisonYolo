from ultralytics import YOLO
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


if __name__==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="train", help="Key params")
    parser.add_argument("--params_file", default="params.yaml", help="Source file with the parameters")
    args = parser.parse_args()
    with open(args.params_file, 'r') as fd:
        params = yaml.safe_load(fd)
    print(args.params)
    model = YOLO("runs/detect/"+params["train"]["save_path"]+"/"+params[args.params]["weights_path"])

    # Display model information (optional)
    model.info()
    # Validate the model
    metrics_test = model.val( data=params["train"]["data"],
                             save_json=True,
                             imgsz=params["train"]["img_size"],
                             batch=params["train"]["batch"],
                             conf=0.25, iou=0.6, device="0", plots=True,
                             name=params["train"]["save_path"]+"/"+params[args.params]["task"],exist_ok=True, split=params[args.params]["task"])  # no arguments needed, dataset and settings remembered
    
     
    metrics_categories = {
         "names":metrics_test.names,
         "map50-95":metrics_test.box.maps,
         "map50":metrics_test.box.ap50,
         "precision":metrics_test.box.p,
         "recall":metrics_test.box.r,
         "f1":metrics_test.box.f1
    }
    metrics = {"map50-95":metrics_test.box.map,  # map50-95
        "map50":metrics_test.box.map50,  # map50
        "map75":metrics_test.box.map75,  # map75
        "precision":metrics_test.box.mp,  # P
        "recall":metrics_test.box.mr, #R
        "fitness": metrics_test.fitness,
        "categories":metrics_categories} 

    json_obj = json.dumps(metrics, indent=4, cls=NumpyArrayEncoder)

    with open("runs/detect/"+params["train"]["save_path"]+"/"+params[args.params]["task"]+"/test_metrics.json", "w") as outfile:
            outfile.write(json_obj)


    """

    results = model("train_dataset/data/BRSNN-CA-2021/RES-S-2021-ZX-1ZE-S-010-CA-CA4-NF2/images/RES-S-2021-ZX-1ZE-S-010-CA-CA4-NF2_001_204_1_M2_CAMERA.JPG")
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
    result.save(filename='resultYolov5.jpg')  # save to disk
    result.save_txt(txt_file='resultYolov5.txt')  # save to disk
    print(result.summary())  # save to disk

    """