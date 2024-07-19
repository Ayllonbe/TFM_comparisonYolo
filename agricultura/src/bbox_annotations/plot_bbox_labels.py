import argparse
from PIL import Image, ImageDraw, ImageColor, ImageFont
from pathlib import Path
from tifffile import TiffFile
import random
import numpy as np
import sys
from os.path import dirname, realpath
curr_path = dirname(realpath(__file__))
curr_path = curr_path.replace('\\','/')
main_path,_ = curr_path.rsplit('/',1)
main_path,_ = main_path.rsplit('/',1)
sys.path.append(main_path)
from src.bbox_annotations.annotations_io import *


#Create the list of colors
PIL_COLORS =  random.sample(list(ImageColor.colormap.keys()), len(ImageColor.colormap.keys()))
CLASS_COLORS = ["red", "cyan", "blue", "fuchsia", "limegreen", "magenta"] + PIL_COLORS
try:
    FONT = ImageFont.truetype("Arial.ttf", 20, encoding="unic")
except:
    FONT = ImageFont.load_default()

def plot_bbox_results(img_PIL, bbox_annotations, colors=CLASS_COLORS, width=5, font = FONT, class_mapping = None):
    # img = Image.fromarray(img_array).convert("RGBA")
    draw = ImageDraw.Draw(img_PIL)
    for bbox in bbox_annotations:
        if bbox is not None:
            class_idx = [str(c) for c in class_mapping] if class_mapping is not None else list(set(np.array(bbox_annotations)[:, -1]))
            [x1, y1, x2, y2, lbl] = bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=colors[class_idx.index(lbl)], width= width)
            fs= font.getbbox(f" {lbl}")
            draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1  + 1], fill=colors[class_idx.index(lbl)])
            draw.text((x1, y1- fs[1]), f" {lbl} ", anchor='ls', fill=(255, 255, 255), font= font )

    return img_PIL


def visualize_bbox_labels(images_path, labels_path, ext=".JPG", target_path='',format="yolo", image_size_resolution=[1.0, 1.0], **kwargs):
    for im_fname, label_fname in zip(images_path.rglob(f'*{ext}'), labels_path.rglob('*.txt')):
        with Image.open(im_fname) as img:
            #image_size_resolution = get_resolution(im_fname) if ext ==".tiff" else [1.0, 1.0]
            image_size_resolution = img.size if format == "yolo" else image_size_resolution
            bbox_labels = read_label(labels_path / Path(im_fname.name).with_suffix(".txt"), image_size_resolution)
            label_img = plot_bbox_results(img, bbox_labels, colors=CLASS_COLORS, width=4, class_mapping = range(kwargs.get('n_classes', 10)) )

            if args.show:
                label_img.show()
            else:
                Path(target_path).mkdir(exist_ok=True, parents=True)
                label_img.save(target_path / Path(im_fname.name).with_suffix('.png'))

def crop_bounding_boxes(filename, bbox_annotations, save_path=''):
    """Crop image specified by filename to coordinates specified."""

    # Open image and get height and width
    image = Image.open(filename)
    bbox_crops = []
    for i,bbox in enumerate(bbox_annotations):
        if bbox is not None:
            [x1, y1, x2, y2, lbl] = bbox
            # Work out crop coordinates, top, left, bottom, right
            bbox_crops.append(image.crop((x1, y1, x2, y2)))
            Path(save_path).mkdir(exist_ok=True, parents=True)
            bbox_crops[i].save(f"{Path(save_path) / Path(filename).stem}_bbox_{i}.{Path(filename).suffix}")
    return bbox_crops


def crop_bbox_labels(images_path, labels_path, ext=".jpg", target_path='',format="yolo", image_size_resolution=[1.0, 1.0], **kwargs):
    for im_fname, label_fname in zip(images_path.rglob(f'*{ext}'), labels_path.rglob('*.txt')):
        with Image.open(im_fname) as img:
            image_size_resolution = img.size if format == "yolo" else image_size_resolution
            bbox_labels = read_label(labels_path / Path(im_fname.name).with_suffix(".txt"), image_size_resolution)
            crop_bounding_boxes(im_fname, bbox_labels, target_path )


def collection_label_visualization(images_path, target_path, n_classes, ext="jpg",):
    for collection in Path(images_path).rglob("*/images"):
        print("PLOTTING IMAGES", collection)
        visualize_bbox_labels(collection, collection.parent / "labels", target_path=target_path,
                              n_classes=n_classes)
        # crop_bbox_labels(coll  ection, collection.parent /"labels", target_path=collection.parent/"bbox_crops", n_classes=args.n_classes)


if __name__==  '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path",  default="annotated_dataset",      help="Source folder with images")
    parser.add_argument("--labels_path",  default=None, help="dataset")
    parser.add_argument("--label_format", default="yolo",   type=str,     help="Source folder with bbox of images")
    parser.add_argument("--target_path",  default="visualize_labels",   type=str,  help="Target folder wto save review images")
    parser.add_argument("--ext",          default="", help="Image extension in a dataset. Default: .png")
    parser.add_argument("--scale_resolution", default=False, help="Boolean to indicate if reading .tiff")
    parser.add_argument("--show", default=False, help="Boolean to show images. If false hte image will be storage in plot_dataset_labels")
    parser.add_argument("--n_classes", default=10, help="Boolean to show images. If false hte image will be storage in plot_dataset_labels")

    args= parser.parse_args()

    # Configure paths
    target_path = Path(args.target_path)
    target_path.mkdir(exist_ok=True, parents=True)
    images_path = Path(args.images_path)
    labels_path = Path(args.labels_path) if args.labels_path is not None else images_path
    ext = args.ext if args.ext!="" else next(images_path.iterdir()).suffix
    ANNOTATIONS_READER = {"yolo":read_annotations_yolo, "imagej": read_annotations_imagej}
    read_label = ANNOTATIONS_READER.get( args.label_format.lower(), read_annotations)
    if Path(args.images_path).exists():
        visualize_bbox_labels(Path(args.images_path), Path(args.labels_path), target_path=Path(args.target_path),
                              ext=args.ext, n_classes=args.n_classes)
    else:
        collection_label_visualization(args.images_path, args.target_path, ext=args.ext, n_classes=args.n_classes)