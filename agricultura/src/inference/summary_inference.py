import sys

import pandas as pd
import torch
import argparse
import glob
import yaml

from more_itertools import ichunked, chunked
from results_utils import *
from results_visualization import *
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO, YOLOWorld
import sys
try:
    from src.image_processing.ImageTiler import conf_image_tiler
    from src.utils.utils_io import load_text_file
    from src.inference.results_utils import save_preds_txt
except ModuleNotFoundError:
    project_path = Path(__file__).parent.parent.parent.as_posix()
    sys.path.append(project_path)
    from src.image_processing.ImageTiler import conf_image_tiler
    from src.utils.utils_io import load_text_file
    from src.inference.results_utils import save_preds_txt

random.seed(0)
if __name__==  '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",         type=str, default="params.yaml", help="Source file with the parameters")
    parser.add_argument("--images_path",    type=str, default="dataset/inference", help="Source Directory with images")
   # Load the parameters file
    args = parser.parse_args()
    with open(args.params, 'r') as fd:
        params = yaml.safe_load(fd)
    results_path = Path("runs/detect/"+params["train"].get("save_path", 'yolo')+"/inference")
    if params["inference"].get("summary", False):
        print("hola")
        summary_df = pd.DataFrame()
        print(f"Results saved in directory '{results_path}'")
        print(glob.glob("runs/detect/"+params["train"].get("save_path", 'yolo')+"/inference/*/*/*.csv"))
        for f in (results_path).glob("*/*/*.csv"):
            results_df = pd.read_csv(f)
            print(f)
            counts = value_counts_df(results_df, "name")
            # summary_df= summary_df.concat(pd.DataFrame(data=counts, index=[f.stem]))
            summary_df = pd.concat( [summary_df,pd.DataFrame(data=counts, index=[f.stem])]  )
        summary_df.to_csv(path_or_buf=str(results_path / "results_summary2.csv"), sep=',', mode='wb')