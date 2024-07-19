import sys

import pandas as pd
import torch
import argparse
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

def predict(image_path, tiler, model, results_path, **kwargs):
    results_list_df = []

    full_image_path = Path(image_path)
    save_path = results_path / full_image_path.stem.split('_')[0] / "output_images"
    save_path.mkdir(exist_ok=True, parents=True)

    save_results_path = save_path.parent / "predictions"
    save_results_path.mkdir(exist_ok=True, parents=True)

    (save_path.parent / "labels").mkdir(exist_ok=True, parents=True)

    # Create all the image
    image_tiles, n_tiles = tiler(full_image_path)
    # Calculate inference time (ms)
    preprocess = 0
    inference = 0
    postprocess = 0
    for b, img_row in enumerate(ichunked(image_tiles, n_tiles[-1])):

        imgs = list(img_row)
        # Inference
        results = model(imgs, imgsz=img_size, conf=kwargs["conf_thres"], iou=kwargs["nms_thresh"], agnostic_nms=True, verbose=False)  # batched inference including NMS

         #Filter results that are on the overlaped stride region
        preprocess = preprocess + results[0].speed["preprocess"]
        inference = inference + results[0].speed["inference"]
        postprocess = postprocess + results[0].speed["postprocess"]
        tile_preds, df = filter_results_yolov5(
                            results, tiler.tile_size(), stride_size=tiler.stride_size(),
                            pad=tiler.pad_size(), batch=b)
        
        results_list_df.append(df)
    
    results_df = pd.concat(results_list_df)

    plot_full_image_bbox(tiler.full_image, results_df, save_path, full_image_path.name)

    save_preds_txt(results_df.values, full_image_path, tiler.full_image.shape, target_path=save_path.parent / "labels")
    results_df.to_csv(path_or_buf=str(save_results_path / full_image_path.stem) + ".csv", sep=',', columns=None, header=True, mode='w')
    return preprocess, inference, postprocess


if __name__==  '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",         type=str, default="params.yaml", help="Source file with the parameters")
    parser.add_argument("--images_path",    type=str, default="dataset/inference", help="Source Directory with images")
   # Load the parameters file
    args = parser.parse_args()
    with open(args.params, 'r') as fd:
        params = yaml.safe_load(fd)
    kwargs = params["inference"]
    device = params["inference"].get('device', 'cpu')
    
    if device == "cpu":
        print( "Perform inference (on CPU))")
        device = ""
    else:
        print("Perform inference - on DEVICE", device)
    if not torch.cuda.is_available(): 
        print("INFERENCE: GPU not available")
    else:
        print("INFERENCE: GPU available")
 
    params["test_data"]["tile_size"] = params["data"]["tile_size"]
    params["test_data"]["tile_stride"] = params["data"]["tile_stride"]
    params["test_data"]["pad_borders"] = params["data"]["pad_borders"]
    params["test_data"]["select_classes"] = params["preprocessing"]["select_classes"]
    # Create the image tiler object
    tiler = conf_image_tiler(params["test_data"])
    img_size = params["data"].get("tile_size", [1280])[0] #@Aaron did change it:  float(params["test_data"].get("tile_size", [1280])[0])
    images_path = params["test_data"].get("images_path", args.images_path)

    # Create the yolov5 model
    kwargs = params["inference"]
    device = params["train"].get('device', 'cpu')
    if "yoloworld" in params["train"]["save_path"] :
        model = YOLOWorld("runs/detect/"+params["train"]["save_path"]+"/"+params["test_data"]["weights_path"])
    else:
        model = YOLO("runs/detect/"+params["train"]["save_path"]+"/"+params["test_data"]["weights_path"])

    class_names = model.names
    results_path = Path("runs/detect/"+params["train"].get("save_path", 'yolo')+"/inference")
    results_path.mkdir(exist_ok=True, parents=True)

    if Path(images_path).is_file():
        fnames = load_text_file(Path(images_path))
        n_files = len(list(fnames).copy())
    else:
        fnames = Path(images_path).rglob('*.*')
        n_files = len(list(Path(images_path).rglob('*.*')))


    # Iterate over all the files in the directory that has the corresponding label
    inference_time = pd.DataFrame(columns=["file","preprocess","inference","postprocess"])
    for im_fname in tqdm(fnames, desc="Predicting Images", total = n_files ):
        preprocess, inferencem, postprocess = predict(im_fname, tiler, model, results_path, **kwargs)
        inference_time.loc[len(inference_time.index)] = [im_fname, preprocess, inferencem, postprocess ]
    inference_time.to_csv(path_or_buf=str(results_path / "inference_summary_time.csv"), sep=',', mode='wb')
    if params["inference"].get("summary", False):
        summary_df = pd.DataFrame()
        print(f"Results saved in directory '{results_path}'")
        for f in (results_path).glob("*/*/*.csv"):
            results_df = pd.read_csv(f)
            counts = value_counts_df(results_df, "name")
            # summary_df= summary_df.concat(pd.DataFrame(data=counts, index=[f.stem]))
            summary_df = pd.concat( [summary_df,pd.DataFrame(data=counts, index=[f.stem])]  )
        summary_df.to_csv(path_or_buf=str(results_path / "results_summary.csv"), sep=',', mode='wb')
