import yaml
from pathlib import Path
import argparse

if __name__==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", default="train_dataset.yaml", help="Source file with the yolov5 datast paths")
    args = parser.parse_args()

    # Read dataset File
    with open(args.dataset_file, 'r') as fd:
        data = yaml.safe_load(fd)

    # Write path
    data['path']=str(Path(data['path']).absolute())

    # Save File
    with open(args.dataset_file, "w") as fw:
        yaml.dump(data, fw, default_flow_style=False, sort_keys=False)