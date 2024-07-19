from ultralytics import YOLO,  YOLOWorld
import argparse
import yaml

if __name__==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="train", help="Key params")
    parser.add_argument("--params_file", default="params.yaml", help="Source file with the parameters")
    args = parser.parse_args()
    with open(args.params_file, 'r') as fd:
        params = yaml.safe_load(fd)
    print(args.params)
    if "yoloworld" in params[args.params]["save_path"] :
         # Load a COCO-pretrained YOLOv5n model
        model = YOLOWorld(params[args.params]["weights"])

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data=params[args.params]["data"],
                            epochs=params[args.params]["epochs"],
                            imgsz=params[args.params]["img_size"],
                            batch=params[args.params]["batch"],
                            exist_ok=True,
                            name=params[args.params]["save_path"]+"/"+params[args.params]["task"],
                            device=params[args.params]["device"],
                            optimizer=params[args.params]["optimizer"],
                            plots=True)
    else:
        # Load a COCO-pretrained YOLOv5n model
        model = YOLO(params[args.params]["weights"])

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data=params[args.params]["data"],
                            epochs=params[args.params]["epochs"],
                            imgsz=params[args.params]["img_size"],
                            batch=params[args.params]["batch"],
                            exist_ok=True,
                            name=params[args.params]["save_path"]+"/"+params[args.params]["task"],
                            device=params[args.params]["device"],
                            optimizer=params[args.params]["optimizer"],
                            plots=True)


