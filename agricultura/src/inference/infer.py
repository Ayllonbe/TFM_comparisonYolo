import numpy as np
import pandas as pd
from yolov5 import hubconf
from yolov5.models.yolo import Model
from yolov5.models.common import Detections
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import random
from image_processing import ImagePatcher


PIL_COLORS =  random.sample(list(ImageColor.colormap.keys()), len(ImageColor.colormap.keys()))
CLASS_COLORS = ["red", "green", "blue", "cyan", "fuchsia", "limegreen", "magenta"] + PIL_COLORS
FONT = ImageFont.load_default()

def load_model(weights_path, device='gpu', autoshape=True):
    """ Laod the yolov5 Hubconf model with qutoshape that includes NMS.  It allows inference from various sources.
        For height=640, width=1280, RGB images example inputs are:
          file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
          URI:             = 'https://ultralytics.com/images/zidane.jpg'
          OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
          PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
          numpy:           = np.zeros((640,1280,3))  # HWC
          torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
          multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

    Parameters
    ----------
    weights_path: str
        path to the trained yolov5 model
    autoshape: bool, optional
        apply autoshape() wrapper to model.
    device: str, torch.device, None,  optional
        device to use for model inference. Choices: cpu,gpu,cuda:0...

    Returns
    -------
        YOLOv5 pytorch evaluation model
    See Also
    --------
        https://github.com/ultralytics/yolov5/issues/36
    """

    model = hubconf.custom(path=weights_path, autoshape=autoshape, device=device)
    model.eval()
    return model


def model_inference(image: Image, yolov5_model: Model, ImageTiler: ImagePatcher) -> pd.DataFrame:
    """
    Helper function to predict Yolov5 model inference. If the image size is bigger that the tile shape defined
    the sliding window approach over the image is computed. See

    Parameters
    ----------
        image: PIL.Image
        yolov5_model: Model
        ImageTiler: Image Patcher Class object
    Returns
    -------
        results:pd.Dataframe
    """
    tile_shape = ImageTiler.tile_size

    if list(image.size) > list(tile_shape):
        results = []
        image_tiles = ImageTiler(np.array(image)).astype(np.uint8)
        for b, img_row in enumerate(image_tiles):
            # Batch Inference by row
            preds = yolov5_model(list(img_row), size=tile_shape[0])  # batched inference including NMS
            # Filter models that are on the overlaped stride region
            preds_df = filter_results_yolov5(preds, tile_shape, stride_size=ImageTiler.stride_size, pad=ImageTiler.pad_shape, batch=b)
            results.append(preds_df)
        # Return the resuts as pandas Dataframe
        results = pd.concat(results)
        result_image = plot_full_image_bbox(ImageTiler.full_image, results )
    else:
        image = image.resize(tuple(tile_shape), Image.ANTIALIAS)
        results = yolov5_model(list(image), size=tile_shape)  # batched inference including NMS
        results = results.pandas().xyxy[0]
        result_image = plot_full_image_bbox(image, results )

    return results, result_image


def in_center_roi(xc: float, yc: float, tile_width: int, tile_height: int, stride_x: int, stride_y: int) -> bool:
    """
    Parameters
    ----------
    xc: float
        center x coordinate of the bounding box
    yc: float
        center y coordinate of the bounding box
    tile_width: int
    tile_height: int
    stride_x: int
    stride_y: int

    Returns
    -------
    in_roi: bool
        indicate if the coordinates are in the region of interest
    """
    center_roi = [stride_x // 2, stride_y // 2, (tile_width - stride_x // 2), (tile_height - stride_y // 2)]
    in_roi = False
    if xc >= center_roi[0] and yc >= center_roi[1] and xc < center_roi[-2] and yc < center_roi[-1]:
        in_roi = True
    return in_roi


def filter_results_yolov5(model_prediction: Detections, tile_size, stride_size=[0, 0], pad=[0, 0], batch=0)-> pd.DataFrame:
    """ Function that filters the prediction if they are in the center ROi based on patch slides
    Parameters
    ----------
    model_prediction: object, Detections
        
    tile_size: list
    stride_size: list, np.ndarray
    pad: list, np.ndarray
    batch: int

    Returns
    ----------
    results: pd.Dataframe
        results containing the filtered detections based on ROI filtering
    """
    tile_width, tile_height = tile_size
    stride_x, stride_y = stride_size
    preds = [None] * len(model_prediction)
    preds_df = []
    indx = []
    results_df = model_prediction.pandas().xyxy
    for i, pred in enumerate(model_prediction.pred):
        p = 0
        preds[i] = [None] * len(pred)
        for det, (r, row) in zip(pred, results_df[i].iterrows()):  # (xyxy, conf, cls)
            xc = ((det[0] + det[2]) / 2).detach().cpu().numpy()
            yc = ((det[1] + det[3]) / 2).detach().cpu().numpy()
            if in_center_roi(xc, yc, tile_width, tile_height, stride_x, stride_y):
                preds[i][p] = row.values  # [x1,y1,x2,y2,conf,cls,name]
                indx.append(f"{str(batch).zfill(2)}-{str(i).zfill(2)}-{str(p).zfill(2)}")

                x1, y1, x2, y2, conf, cls, name = row.values
                x1 = x1 + i * (tile_size[0] - stride_size[0]) - pad[0] // 2
                x2 = x2 + i * (tile_size[0] - stride_size[0]) - pad[0] // 2
                y1 = y1 + batch * (tile_size[1] - stride_size[1]) - pad[1] // 2
                y2 = y2 + batch * (tile_size[1] - stride_size[1]) - pad[1] // 2
                preds_df.append([x1, y1, x2, y2, conf, cls, name])
                p += 1

    return pd.DataFrame(preds_df, index=indx, columns=results_df[0].columns)


def draw_bbox(img: Image, bbox_preds:np.ndarray,  colors =CLASS_COLORS):
    """Draw the prediciton bounding box (including conf value and label name) on PIL draw image object

    Parameters
    ----------
    img : PIl.Image
    bbox_preds: pd.Dataframe
    colors: dict, optional
        dict containing the PIL colors for drawing the bounding box for each class
    """
    draw = ImageDraw.Draw(img)
    for bbox in bbox_preds:
        if bbox is not None:
            x1, y1, x2, y2, conf, lbl, name = bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=colors[lbl], width=3)
            fs = FONT.getsize(f"{ name} {conf:.2f}")
            draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1 + 1], fill=colors[lbl])
            draw.text((x1, y1- fs[1]), f"{ name} {conf:.2f}", anchor='ls', fill=(255, 255, 255), font=FONT)
    return img


def plot_full_image_bbox(image: np.ndarray, bbox_preds_df:pd.DataFrame, **kwargs):
    """ Function that draws the filtered prediciton Bounding boxes to the full inference image "
    Parameters
    ----------
    image: np.ndarray
    bbox_preds_df: pd.Dataframe

    Returns
    ----------
    image: PIL.Image
        results the full inference image
    """

    colors = kwargs.get("colors", CLASS_COLORS)
    image_PIL = Image.fromarray(image).convert("RGB")
    image_PIL = draw_bbox(image_PIL,bbox_preds_df.values, colors)
    return image_PIL
