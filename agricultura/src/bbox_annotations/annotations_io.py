
from pathlib import Path
from tifffile import TiffFile
import numpy as np
from src.utils.utils_io import load_json
import logging

def save_labels_yolo(bboxes, label_fname, image_size,  class_labels = {} , scale=[1.0, 1.0] ):
    # Yolo Label Format:   [class_id, x_center, y_center, width_normalized, height_normalized]
    with open(label_fname, 'w', encoding='utf-8') as label_output_file:
        for bbox in bboxes:
            x_center = float( (bbox[1] + bbox[3])  /  (2*image_size[0]))
            y_center = float((bbox[2] + bbox[4])   /  (2*image_size[1]))
            width = float( abs(bbox[3] - bbox[1])  / (image_size[0]))
            height = float(abs(bbox[4] - bbox[2])  / (image_size[1]))

            # Reduce the bbox_size by a factor
            width =  float(width  *  scale[0])
            height = float(height * scale [1])

            # YOLO format need the label index, order given by the class_names args
            lbl = int(class_labels.get(bbox[0], bbox[0]) ) # lbl = int(self.class_names.index(bbox[0]))

            # Write the args
            label_output_file.write( f"{lbl} {x_center} {y_center} {width} {height}\n")

def read_annotations(filename, label_format, image_size_resolution):
    """read annotation from annotation text file (.txt) saved as yolo """
    # Check in case of error if is in the same dir than the annotation
    label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
    bboxes = []
    with open(label_file, 'r') as fin:
        for line in fin:
            anno = line.rstrip('\n').split(' ')
            label = anno[0].split(":")[0] # in case of imageJ

            if label_format.lower() =="yolo":
                x_center = float(anno[1]) * image_size_resolution[0]
                y_center = float(anno[2]) * image_size_resolution[1]
                width = float(anno[3]) * image_size_resolution[0]
                heigth = float(anno[4]) * image_size_resolution[1]

                x1 = int(x_center - width/2)
                x2 = int(x_center + width/2)
                y1 = int(y_center - heigth/2)
                y2 = int(y_center + heigth/2)

            elif label_format.lower() =="coco":
                x_center =  float(anno[1])
                y_center =  float(anno[2])
                width =     float(anno[3])
                heigth =    float(anno[4])

                x1 = int(x_center - width/2)
                x2 = int(x_center + width/2)
                y1 = int(y_center - heigth/2)
                y2 = int(y_center + heigth/2)

            elif label_format.lower() =="imagej":
                # Resolution must coincide with pixel resolution

                x1 = int(float(anno[4]))* image_size_resolution[0]
                y1 = int(float(anno[5]))* image_size_resolution[0]
                x2 = int(float(anno[6]))* image_size_resolution[0]
                y2 = int(float(anno[7]))* image_size_resolution[0]

            else:
                x1 = int(anno[1])
                y1 = int(anno[2])
                x2 = int(anno[3])
                y2 = int(anno[4])

            bboxes.append( [x1,y1,x2,y2,label] )

    return bboxes

def read_annotations_yolo(filename, image_size):
    """read annotation from annotation text file (.txt) saved as yolo """
    # Check in case of error if is in the same dir than the annotation
    label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
    bboxes = []
    with open(label_file, 'r') as fin:
        for line in fin:
            anno = line.rstrip('\n').split(' ')
            label = anno[0]
            x_center =  float(anno[1])*image_size[0]
            y_center =  float(anno[2])*image_size[1]
            width =     float(anno[3])*image_size[0]
            heigth =    float(anno[4])*image_size[1]

            x1 = int(x_center - width/2)
            x2 = int(x_center + width/2)
            y1 = int(y_center - heigth/2)
            y2 = int(y_center + heigth/2)

            bboxes.append( [x1,y1,x2,y2,label] )

    return bboxes

def read_annotations_coco_txt(filename, image_size=[640, 640]):
    """read annotation from annotation text file (.txt) saved as coco """
    # TODO: DOCUMENT
    coco_json = load_json(filename)

    # Create image dictionary by image_id
    images_dict = {'%s' % Path(x['file_name']).name: x for x in coco_json['images']}
    images_fnames = [ x['file_name'] for x in coco_json['images']]
    labels = dict(keys=[ x['file_name'] for x in coco_json['images']])
    # Write labels file
    for x in coco_json['bbox_annotations']:
        img = images_dict['%g' % x['image_id']]
        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        cls = x['category_id'] - 1      # class
    return labels

    # Check in case of error if is in the same dir than the annotation
    label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
    bboxes = []
    with open(label_file, 'r') as fin:
        for line in fin:
            anno = line.rstrip('\n').split(' ')
            label = anno[0]
            x_center =  float(anno[1])
            y_center =  float(anno[2])
            width =     float(anno[3])
            heigth =    float(anno[4])

            x1 = int(x_center - width/2)
            x2 = int(x_center + width/2)
            y1 = int(y_center - heigth/2)
            y2 = int(y_center + heigth/2)

        bboxes.append( [x1,y1,x2,y2,label] )

    return bboxes

def read_annotations_imagej(filename, image_resolution=[1.0, 1.0]):
    """read annotation from annotation text file (.txt) saved as yolo """
    # Check in case of error if is in the same dir than the annotation
    label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
    bboxes = []
    with open(label_file, 'r') as fin:
        for line in fin:
            anno = line.rstrip('\n').split(' ')
            label = anno[0].split(":")[0]
            x1 = int(float(anno[4])*image_resolution[0])
            y1 = int(float(anno[5])*image_resolution[1])
            x2 = int(float(anno[6])*image_resolution[0])
            y2 = int(float(anno[7])*image_resolution[1])
            bboxes.append( [x1,y1,x2,y2,label] )

    return bboxes.copy()


def get_resolution(image_path):
    resolution = [1.0,1.0]
    if image_path.suffix in [".tiff", ".tif"]:
        image = TiffFile(str(image_path))
        try: # Try to extract resolution from Tiff image
            resolution[0] = image.pages[0].tags['XResolution'].value[0] / image.pages[0].tags['XResolution'].value[1]
            resolution[1] = image.pages[0].tags['YResolution'].value[0] / image.pages[0].tags['YResolution'].value[1]
        except:
            logging.error(  f"{str(image_path.name)}: meta-data could not be read or resolution info is not available")
    return resolution

