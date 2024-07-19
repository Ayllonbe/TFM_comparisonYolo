from pathlib import Path
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import random
from src.image_processing import ImagePatcher
try:
    FONT = ImageFont.truetype("yolov5_cfg/Arial.ttf", 20, encoding="unic")
except:
    FONT = ImageFont.load_default()
random.seed(0)
#Create the list of colors
PIL_COLORS =  random.sample(list(ImageColor.colormap.keys()), len(ImageColor.colormap.keys()))
CLASS_COLORS = ["red", "green", "blue", "cyan", "fuchsia", "limegreen", "magenta"] + PIL_COLORS

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


def plot_full_image_bbox(image: np.ndarray, bbox_preds_df:pd.DataFrame, target_path, im_fname,**kwargs):
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
    image_PIL.save(target_path/Path(im_fname).with_suffix(".png"))
    del image_PIL



def plot_tile_bbox_results(image_tiles, bbox_preds_list, target_path, im_fname, stride=[0, 0],batch = 0, **kwargs):
    "Draws the prediciton drawn Bounding boxes for each predicted tile "
    colors = kwargs.get("colors", class_colors)
    overlap =   (kwargs.get("crop_overlap",4))//2
    for i, tile in enumerate(image_tiles):
        img = Image.fromarray(tile).convert("RGBA")
        if bbox_preds_list[i]!=[]:
            img = draw_bbox(img,bbox_preds_list[i], colors)
        # Crop the tiles to only display the non Overlapping region
        width, height = img.size
        img = img.crop((stride[0]//overlap, stride[1]//overlap, width- stride[0]//overlap, height - stride[1]//overlap ))
        img.save(target_path/Path(im_fname + f"-tile-{ str(batch).zfill(2)}-{str(i).zfill(2)}").with_suffix(".png"))

def plot_multilayer_image_bbox(image, bbox_preds_df, target_path, im_fname,colors=CLASS_COLORS, n_layers = [4,5],  tile_shape = (1280,1280)):
    "Draws the prediciton drawn Bounding boxes to the multilayer full image"

    frames=[]
    image_shape = image.shape
    dtype = image.dtype
    for i, page in enumerate(image[n_layers[0]:n_layers[1]]):
        page = Image.fromarray(page)
        draw = ImageDraw.Draw(page)
        for p,bbox in bbox_preds_df.iterrows():
            draw_bbox(draw,bbox, colors)
        frames.append(np.array(page.im))
    del image
    with tifffile.TiffWriter(target_path/Path(im_fname).with_suffix(".tiff"), bigtiff=True) as tif:
        tif.write( data=tif_tiler(frames,(len(frames),) + image_shape[1:],  tile_shape), shape=(len(frames),) + image_shape[1:],
                   dtype=dtype, tile=tile_shape)
