from pathlib import Path
import argparse

import PIL.Image
import numpy as np
import rawpy
import imageio
import logging
import tifffile
from tifffile import TiffFile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter
import yaml
import cv2
import mahotas as mh
import sys, os
curr_path = os.path.dirname(os.path.realpath(__file__))
curr_path = curr_path.replace('\\','/')
main_path,_ = curr_path.rsplit('/',1)
main_path,_ = main_path.rsplit('/',1)
sys.path.append(main_path)

from src.utils.utils_io import load_text_file

IMAGE_EXT =['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif', '.ppm', '.xbm', '.tiff', '.rgb', '.pgm', '.png', '.pnm', '.cr2','.nef', '.x3f', '.kdc']

def set_logs(log_level):
    _logLevel = {
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARNING,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    level = _logLevel.get(log_level.upper(),logging.NOTSET)
    # Configure and create logger
    logging.basicConfig(filename="preprocessing.log", format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    # Setting the threshold of logger
    logger.setLevel(level)
    return logger


class BboxImagePreprocessor(object):

    def __init__(self, tile_size, tile_strides, save_path = None, **kwargs):

        self.tile_size = np.array(tile_size) if len(tile_size)>=2 else  np.array([tile_size, tile_size])
        self.tile_stride = np.array(tile_strides)[::-1] if len(tile_strides)>=2 else np.array([tile_strides, tile_strides]) # Flip coordinates to match numpy convention
        self.PADDING = kwargs.get("padding", True )
        self.pad = 0
        self.pad_color = kwargs.get("pad_color", [0,0,0] ) #RGB code (0-255) for padded pixels black [0,0,0]

        self.ext = kwargs.get("ext", ".png" ).lower()
        self.ext = self.ext if "." in self.ext else "." + self.ext

        self.save_ext = kwargs.get("save_format", self.ext) # if not specified, same format as original
        self.save_root_path = Path(save_path) if save_path is not None else None
        if self.save_root_path is not None:  # If saving directory is specified, create them
            print("Preprocess Images will be saved in ", self.save_root_path)
            Path(self.save_root_path / "images").mkdir(parents=True, exist_ok=True)
            Path(self.save_root_path / "labels").mkdir(parents=True, exist_ok=True)

        self.class_names = kwargs.get("class_names", None)
        self.class_count = Counter()

        #Select tto save the tiled labels as yolo format or COCO in txt
        # self.save_labels = self.save_labels_yolo if  kwargs.get("label_format", "yolo") == "yolo" else self.save_labels_coco
        if  kwargs.get("label_format", "yolo") == "yolo":
            self.save_labels = self.save_labels_yolo
            self.read_labels=self.read_annotations_yolo
            # Create default class labels as num index
            #self.class_labels = kwargs.get("new_class_labels", dict(zip(self.class_names, range(len(self.class_names))))) # self.class_label2index(self.class_names))
            self.class_labels = dict(zip(self.class_names, kwargs.get("new_class_labels", range(len(self.class_names)) ) ) )
        else:
            self.read_labels=self.read_annotations_imagej2bbox
            self.save_labels = self.save_labels_coco
            self.class_labels = dict( zip( self.class_names, kwargs.get("new_class_labels", self.class_names) ) )
        self.balance_classes = kwargs.get("balance_data", False)   #Oversample underrepresented classes
        self.multichannel = kwargs.get("multichannel", False )
        if self.ext.lower() in ['.cr2','.nef', '.x3f', '.kdc' ]:
            self.loader = self.load_raw_image
        if  self.ext.lower() in ['.ndpi','.svs', '.tiff', '.tif'] and self.multichannel:
            self.loader = self.load_microscopy_image_multilayer
        elif self.multichannel:
            self.loader = self.load_multichannel_image
        else:
            self.loader = self.load_image_PIL

        self.read_resolution = (kwargs.get("read_resolution", False) and self.ext.lower() in [".tiff", ".tif"])
        self.bbox_resolution = kwargs.get("bbox_resolution",  [1.0, 1.0])

        merge_strategies = {"min": self.merge_min, "median": self.merge_median,"mean": self.merge_mean,
                            "std": self.merge_std,  "focus":  self.focus,
                            "sharpest_canny": self.select_channel_layer, "sharpest": self.determine_sharpest_layer}

        if self.multichannel :
            merge_ch = kwargs.get("merge_channels", False).lower()
            print("Merge layers with ", merge_ch)
            self.merge_channels = merge_strategies.get(merge_ch,self.create_multichannel)
            self.n_layer = kwargs.get("n_layers", 1 ) -1

        # Select image writer if multichannel volume, else write 2D image
        self.img_writer = imageio.volwrite if (self.multichannel and self.save_ext.lower() in [".tiff", ".tif"] ) else imageio.imwrite


    def __call__(self, image_path, label_path, *args, **kwargs):

        # Read the multichannel image
        image = self.loader(image_path)

        # The resolution divided by 4 is the correction factor from the 4x4 tiled ndpi dataset
        #image_dims = np.array(self.get_resolution(image_path)) if self.read_resolution else list(image.shape)[:-1]
        height, width = np.array(self.get_resolution(image_path)) if self.read_resolution else list(image.shape)[:-1]

        # Read the annotation file as [x1,y1,x2,y2,label]
        bbox_annotations = self.read_labels(label_path, self.class_names, [width , height])

        # Count the total labels sample by class for all images
        self.class_count.update(self.count_class_labels(bbox_annotations))

        if self.PADDING: #pad the borders
            offset = self.__offset(image.shape[-3:-1])
            self.pad =  self.tile_stride - offset if  offset.all() > 0 else np.zeros_like( self.tile_stride)
            image = self.__pad_borders(image, self.pad , color = self.pad_color )

        # Tile image dimensions (numpy flips height and width shapes)
        n_rows =     np.array(image.shape[-2]) // self.tile_stride[1] if  self.tile_stride[1] == self.tile_size[1] else np.array(image.shape[-2]) // self.tile_stride[1] -1
        n_columns  = np.array(image.shape[-3]) // self.tile_stride[0] if  self.tile_stride[0] == self.tile_size[0] else np.array(image.shape[-3]) // self.tile_stride[0] -1

        #Create empty list for storing the tile images and labels
        tile_images = [None] * (n_rows* n_columns)
        tile_labels = [None] * (n_rows* n_columns)
        tile_fnames = [None] * (n_rows* n_columns)

        for r in range(n_rows):
            for c in range(n_columns):
                i = r * n_columns + c
                tile_images[i] = next(self.__tile(image, r, c ))
                tile_labels[i] = next(self.__create_tile_bbox(bbox_annotations, r, c ))
                tile_fnames[i] = (Path(image_path).name).replace('.', f'_x{str(c).zfill(2)}_y{str(r).zfill(2)}.')

                if self.save_root_path is not None:
                    self.save_image( tile_images[i], Path(tile_fnames[i]).name)
                    self.save_labels(tile_labels[i], self.save_root_path / "labels" / Path(tile_fnames[i]).with_suffix(".txt"))

        if self.balance_classes and (self.save_root_path is not None):
            self._create_oversample_images(bbox_annotations, image, image_path, tile_fnames, tile_images, tile_labels)

        # Return the list generators
        return (tile_images, tile_labels, tile_fnames)

    def _create_oversample_images(self, bbox_annotations, image, image_path, tile_fnames, tile_images, tile_labels):
        """ Helper function that creates oversampled image object """
        labels = [bbox[-1] for bbox in bbox_annotations]
        count = self.count_class_labels(bbox_annotations)
        logging.info(f"The class label count for image {Path(image_path).name} is: {count}")
        for cls in np.unique(labels):  # self.class_names: #select only the classes contained in that image
            # Select the underrepresented classes index the difference times
            ids_oversample = np.random.choice([i for i, c in enumerate(labels) if c == cls],
                                              max(count.values()) - count[cls])
            for i, id in enumerate(ids_oversample):
                x1, y1, x2, y2, label = bbox_annotations[id]
                x_center = int(
                    (x1 + x2) / 2 + np.random.uniform(low=-self.tile_size[0] // 10, high=self.tile_size[0] // 10))
                y_center = int(
                    (y1 + y2) / 2 + np.random.uniform(low=-self.tile_size[1] // 10, high=self.tile_size[1] // 10))
                balance_image = next(self.__tile_from_xy(image, x_center, y_center))
                balance_label = next(self.__create_bbox_from_xy(bbox_annotations, x_center, y_center))
                balance_image, balance_label = self.__remove_object(balance_image, balance_label, cls)
                if self.multichannel and self.merge_channels:  balance_image = self.multichannel_fusion(balance_image)
                balance_fname = (Path(image_path).name).replace('.',
                                                                f'_x{str(x_center // self.tile_size[0]).zfill(2)}_y{str(y_center // self.tile_size[1]).zfill(2)}_bal_{i}_{cls}.')
                if balance_image.size != 0 and (self.save_root_path is not None):
                    self.save_image(balance_image, Path(balance_fname).name)
                    self.save_labels(balance_label, self.save_root_path / "labels" / Path(balance_fname).with_suffix(".txt"))
                    tile_images.append(balance_image), tile_labels.append(balance_label), tile_fnames.append(
                        balance_fname)
                    logging.info(f"Balanced annotations {Path(balance_fname).name} created")

    def __tile(self, image, tile_x, tile_y):
        # Tile Image generator for the given full image
        idx =+ tile_x * self.tile_stride[0]
        idy =+ tile_y * self.tile_stride[1]
        yield image[..., idy:idy+self.tile_size[1],idx:idx+self.tile_size[0], :]


    def __tile_from_xy(self, image, x_center, y_center):
        # Tile image generator for the given (x,y)
        idx = max( int(x_center) - (self.tile_size[0])//2, 0)
        idy = max( int(y_center) - (self.tile_size[1])//2, 0)

        yield image[..., idy:idy+self.tile_size[1],idx:idx+self.tile_size[0], :]

    def __create_bbox_from_xy(self, bboxes,  x_center, y_center):
        idx = max( int(x_center) - (self.tile_size[0])//2, 0)
        idy = max( int(y_center) - (self.tile_size[1])//2, 0)
        bboxes_tiles = []
        for x1, y1, x2, y2, label in bboxes:
            cx = (x2 - x1)//2  + x1 - idx
            cy = (y2 - y1)//2  + y1 - idy
            # If the center of the bbox lies within the tile
            if ((cx >= 0 and cx < self.tile_size[0]) and
                    (cy >= 0 and cy < self.tile_size[0])):
                x1 = x1 - idx if x1 -idx > 0 else 0 # Top left corner
                y1 = y1 - idy if y1 - idy > 0 else 0 # Top left corner
                x2 = x2 - idx if x2 - idx <  self.tile_size[0] else self.tile_size[0] # Top left corner
                y2 = y2 - idy if y2 - idy <  self.tile_size[1] else self.tile_size[1] # Top left corner
                bboxes_tiles.append( [label, x1, y1, x2, y2])

        yield bboxes_tiles

    def __create_tile_bbox(self, bboxes, tile_x, tile_y):
        idx =+ tile_x * self.tile_stride[0]
        idy =+ tile_y * self.tile_stride[1]
        bboxes_tiles = []

        for x1, y1, x2, y2, label in bboxes:
            cx = (x2 - x1)//2  + x1 - idx
            cy = (y2 - y1)//2  + y1 - idy
            # If the center of the bbox lies within the tile
            if ((cx >= 0 and cx < self.tile_size[0]) and
                    (cy >= 0 and cy < self.tile_size[1])):
                x1 = x1 - idx if x1 -idx > 0 else 0 # Top left corner
                y1 = y1 - idy if y1 - idy > 0 else 0 # Top left corner
                x2 = x2 - idx if x2 - idx <  self.tile_size[0] else self.tile_size[0] # Top left corner
                y2 = y2 - idy if y2 - idy <  self.tile_size[1] else self.tile_size[1] # Top left corner
                bboxes_tiles.append( [label, x1, y1, x2, y2])

        yield bboxes_tiles


    def __remove_object(self, tiled_image, bboxes, keep_class):
        bbox_object =[]
        balance_tiled_image = np.copy(tiled_image)
        for label, x1, y1, x2, y2 in bboxes:
            if label==keep_class:
                bbox_object.append([label, x1, y1, x2, y2])
            else:
                balance_tiled_image[...,  y1:y2, x1:x2, : ] = self.pad_color

        return balance_tiled_image, bbox_object


    def classnames2labels(self, cls_name):
        # Auxiliary method to get the new classa label form the map or return the class labels
        return self.class_labels.get(cls_name, cls_name)

    def save_labels_yolo(self, bboxes, label_fname):
        # Yolo Label Format:   [class_id, x_center, y_center, width_normalized, height_normalized]
        with open(label_fname, 'w', encoding='utf-8') as label_output_file:
            for bbox in bboxes:
                x_center = float( (bbox[1] + bbox[3])  /  (2*self.tile_size[0]))
                y_center = float((bbox[2] + bbox[4])   /  (2*self.tile_size[1]))
                width = float( abs(bbox[3] - bbox[1])  / (self.tile_size[0]))
                height = float(abs(bbox[4] - bbox[2])  / (self.tile_size[1]))

                # Reduce the bbox_size by a factor
                width =  float(width *  self.bbox_resolution[0])
                height = float(height * self.bbox_resolution[1])
                # YOLO format need the label index, order given by the class_names args
                lbl = self.class_labels.get(bbox[0], -1) # lbl = int(self.class_names.index(bbox[0]))
                if lbl>=0:
                    # Write the labels
                    label_output_file.write( f"{lbl} {x_center} {y_center} {width} {height}\n")


    def get_class_count(self):
        return self.class_count

    def save_labels_coco(self, bboxes, label_fname):
        # COCO Label Format: [class_label, [class_label, bbx_upper_left_corner_x, bbx_upper_left_corner_y, width, height]
        with open(label_fname, 'w', encoding='utf-8') as label_output_file:
            for bbox in bboxes:
                x_center = int((bbox[1] + bbox[3])/2)
                y_center = int((bbox[2] + bbox[4])/2)
                width = int(abs(bbox[3] - bbox[1]))
                height = int(abs(bbox[4] - bbox[2]))
                lbl = self.class_labels.get(bbox[0], bbox[0])
                label_output_file.write( f"{lbl} {x_center} {y_center} {width} {height}\n")


    @staticmethod
    def load_image(image_path):
        return imageio.imread(image_path)


    @staticmethod
    def load_image_opencv(image_path):
        return cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)

    @staticmethod
    def load_image_PIL(image_path):
        return np.array(PIL.Image.open(str(image_path)))

    @staticmethod
    def load_raw_image(image_path):
        raw_image = rawpy.imread(str(image_path))
        return  np.array( raw_image.postprocess() )

    @staticmethod
    def load_multichannel_image(image_path):
        return imageio.volread(image_path)

    @staticmethod
    def load_microscopy_image_singlechannel(image_path, level =0 ) :
        import openslide
        slide = openslide.OpenSlide(str(image_path))
        w,h  = slide.level_dimensions[level]
        return np.array(slide.get_thumbnail((w,h)))

    @staticmethod
    def load_microscopy_image_multilayer(image_path):
        image = np.array(tifffile.imread(image_path))
        if len(image.shape) < 4: # fix in case of single layer images present
            image = image[np.newaxis, ...]
        return image

    @staticmethod
    def load_img_metadata(image_path, level=0):
        images=[]
        tags=[]
        with tifffile.TiffFile(image_path) as tif:
            for layer, page  in zip( tif.series[level], tif.pages ):
                images.append(layer.asarray())
                for tag in page.tags:
                    tags.append( [tag.name, tag.value] )

        return images, tags


    @staticmethod
    def count_class_labels(bbox_annotations):
        # Select the label from annotation and flatten the lists
        labels = [bbox[-1] for bbox in bbox_annotations]
        # Count the labels for each class
        class_count = dict((l, labels.count(l)) for l in labels)  # dict(collections.Counter(labels))
        return  class_count

    @staticmethod
    def __pad_borders(image, pad, color = [0, 0, 0], center = False):
        im_size = tuple( image.shape[-3:-1] + pad)
        padded_shape = image.shape[:-3] + tuple( im_size) + (image.shape[-1], )
        # Create padded image as filled with color (RGB) values
        padded_image = np.full(shape=padded_shape, fill_value = color, dtype=image.dtype)

        if center:
            padded_image[...,pad[0]//2:(pad[0]//2)+image.shape[-3],  pad[1] // 2:( pad[1] // 2) +image.shape[-2], :] = image
        else: # Pad only on right bottom boarder not to modify bounding box coordinates
            padded_image[...,:image.shape[-3], :image.shape[-2], :] = image
        return padded_image

    def __offset(self, image_size):
        """  Auxiliary methods to compute the offset for make tilling exactly divisible"""
        image_size = np.array(image_size)
        offset = (image_size) - (self.tile_stride * ((image_size - self.tile_size) // self.tile_stride ) + self.tile_size)
        return offset

    def save_image( self, image, filename, **kwargs):
        slice_path = self.save_root_path / "images" / Path(filename).with_suffix(f'{self.save_ext}') #filename.replace(f'{self.ext}', f'{self.save_ext}')
        self.img_writer(str(slice_path), image, **kwargs)


    @staticmethod
    def read_annotations_imagej2bbox(filename, selected_classes, resolution=[1.0, 1.0]):
        """read annotation from ImageJ generated annotation text file (.txt)"""
        res_x, res_y = resolution
        # Check in case of error if is in the same dir than the annotation
        label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
        bboxes = []
        with open(label_file, 'r') as fin:
            for line in fin:
                anno = line.rstrip('\n').split(' ')
                label = anno[0].split(":")[0]
                # Append label only if label within selected classes
                if label in selected_classes:
                    x1 = int(float(anno[4]) * res_x)
                    y1 = int(float(anno[5]) * res_y)
                    x2 = int(float(anno[6]) * res_x)
                    y2 = int(float(anno[7]) * res_y)
                    assert (x2-x1)*(y2-y1)>=0, "The bounding box annotation is not valid"
                    bboxes.append( [x1,y1,x2,y2,label] )
                else:   logger.warning(f"LABEL {label} NOT IN VALID LIST: " + str(selected_classes))
        return bboxes



    @staticmethod
    def read_annotations_yolo(filename,  selected_classes, image_size):
        """read annotation from annotation text file (.txt) saved as yolo """
        # Check in case of error if is in the same dir than the annotation
        label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
        bboxes = []
        with open(label_file, 'r') as fin:
            for line in fin:
                anno = line.rstrip('\n').split(' ')
                label = anno[0]
                if label in selected_classes:
                    x_center =  float(anno[1]) * image_size[0]
                    y_center =  float(anno[2]) * image_size[1]
                    width =     float(anno[3]) * image_size[0]
                    heigth =    float(anno[4]) * image_size[1]

                    x1 = int(x_center - width/2)
                    x2 = int(x_center + width/2)
                    y1 = int(y_center - heigth/2)
                    y2 = int(y_center + heigth/2)

                    # assert (x2-x1)*(y2-y1)>=0, "The bounding box annotation is not valid"
                    bboxes.append( [x1,y1,x2,y2,label] )
                else:   logger.warning(f"LABEL {label} NOT IN VALID LIST: " + str(selected_classes))

        return bboxes

    @staticmethod
    def class_label2index(class_names):
        label2idx = {}
        for i, c in enumerate(class_names):
            label2idx[c] = i
        return label2idx

    @staticmethod
    def __new_image_size( image_size,  stride, offset):
        """
        Compute new image size
        """
        return image_size - offset + stride + 1

    @staticmethod
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


    def multichannel_fusion(self, image):
        rgb_image = self.merge_channels(image)
        try: # Normalize the image
            rgb_image = ( rgb_image / rgb_image.max() * 255).astype(np.uint8)
        except ValueError:  #raised if rgb_image is empty array
            logging.error(f"The tiled array empty and not saved")
        return rgb_image

    @staticmethod
    def get_canny_thr(image, percent=0.33):

        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower_thr = int(max(0, (1.0 - percent) * v))
        upper_thr = int(min(255, (1.0 + percent) * v))

        # return the automatic thresholds
        return lower_thr, upper_thr

    @staticmethod
    def get_edge_sharpness(image, trh_1=20, trh_2=175):
        """returns sharpness of each image as the average  Canny edge detection algorithm from opencv2
        ------- params ------
            image = 2D array
            trh_1 and thr_2 are the tresshold values which can be tweaked to define the sensitivity of the edge detection.
        """
        canny = cv2.Canny(image, trh_1, trh_2)
        return np.mean(canny)

    @staticmethod
    def save_img(img, folder='preprocessed_data/images', name=None):
        """   Save opencv RGB images under the specified folder """
        cv2.imwrite(f"{folder}/{name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def merge_min(image):
        rgb_img = np.min(image, axis=0)
        return rgb_img
    @staticmethod
    def merge_mean(image):
        rgb_img = np.mean(image, axis=0)
        return rgb_img

    @staticmethod
    def merge_median(image):
        rgb_img = np.median(image, axis=0)
        return rgb_img

    @staticmethod
    def merge_std(image):
        rgb_img = np.std(image, axis=0)
        return rgb_img

    def select_channel_layer(self, image):
        rgb_img = np.array(image)[self.n_layer,...]
        return rgb_img

    @staticmethod
    def determine_sharpest_layer(image,  **kwargs):
        """Determine sharpest layer by that a less sharp image blurred,
        will have less  pixel-values difference thatn a blurred sharp images """
        kernel_size = kwargs.get("kernel_size", (5, 5))
        blur_diff = []
        try:
            for layer_image in image:
                blur_layer = cv2.blur(np.array(layer_image), kernel_size)
                blur_diff.append( np.sum(np.abs( blur_layer - layer_image)))
            # Select the sharpest image as index of the sum
            idx = np.argmax(blur_diff)
        except cv2.error:
            idx = 0
        return image[idx]

    def select_sharpest_layer(self, image):
        # Convert images to rgb
        images_gray = [cv2.cvtColor(img_layer, cv2.COLOR_BGR2GRAY) for img_layer in image]

        # Compute Canny values thresholds Automatically
        trh_1, trh_2 = self.get_canny_thr(images_gray, percent=0.33)

        edge_sharpness = [self.get_edge_sharpness(img_layer, trh_1, trh_2) for img_layer in images_gray ]

        # Get the shorted indices of sharpest layers
        idx_sharp = np.argsort(edge_sharpness)[-self.n_layer:]

        # Select the sharpest n_layers
        image = np.array(image)[idx_sharp,...]

        return image

    def focus(self, image):
        # Implementation of  Extended Depth of Field from mahotas library (https://mahotas.readthedocs.io/en/latest/edf.html)
        focus_img = np.zeros_like(image[0])
        layers, h, w, ch = image.shape
        for c in range(0, ch):
            image_ch = image[:, :, :, c]
            focus = np.array( [mh.sobel(t, just_filter=True) for t in image_ch])
            best = np.argmax(focus, 0)
            # Rehsape the image to avoid nested loops:  (equivalent to)
            # [r[y,x] = image[best[y,x], y, x] for x in xrange(w) for y in xrange(h)]
            image_ch = image_ch.reshape((layers, -1)).transpose()
            # Select the right pixel at each location
            r = image_ch[np.arange(len(image_ch)), best.ravel()]
            # reshape to get final result
            focus_img[:, :, c] = r.reshape((h, w))

        return focus_img.astype(np.uint8)

    def create_multichannel(self, image):
        img_multichannel = np.array(image)[0:self.n_layer,...]
        return img_multichannel


def get_preprocessing_object(**kwargs):

    # Get class names and separate by comma
    class_names = kwargs.get("select_classes", str(["0,1,2,3,4,5,6,7,8,9,10"])).split(",")

    # Extract the image extension from the folder if not specified

    # Configure Tiling image options or default values
    tile_size   = kwargs.get("tile_size", [1280,1280])
    tile_stride = kwargs.get("tile_stride", tile_size)

    # remove ambiguous values
    kwargs.pop("tile_size", None)
    kwargs.pop("tile_stride", None)
    kwargs.pop("select_classes", None)

    # Create image tiler object
    return BboxImagePreprocessor(tile_size=tile_size, tile_strides=tile_stride, class_names=class_names,**kwargs)

def tile_bbox_dataset(fnames,bbox_images_preprocessor, workers=4):
    if workers > 1:  # Load the data with multiple threads
        logging.info(f"Number of workers  {workers}")
        # Use threads to speed up I/O operations
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(lambda p: bbox_images_preprocessor(*p), fnames),
                                desc="Preprocessing Images"))  # pool.map(lambda p: tiler(*p), fnames)

    else:
        # Iterate over all the files in the directory that has the corresponding label
        for im_fname, label_fname in tqdm(zip(fnames), desc="Preprocessing Images", total=len(list(fnames))):
            tile_images, tile_labels, tile_fnames = bbox_images_preprocessor(im_fname, label_fname)


def args_parser(*args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", default="data/images", help="Source folder with images  to be tiled")
    parser.add_argument("--labels_path", default="data/labels", help="Source folder with images  to be tiled")
    parser.add_argument("--target_path", default="preprocess_data",   help="Target folder for a new sliced dataset")
    parser.add_argument("--ext", default=".tiff", help="Image extension in a dataset. Default: .JPG")
    parser.add_argument("--save_format", default=".tiff", help="Image extension in a dataset. Default: .png")
    parser.add_argument("--tile_stride", type=list, default=[640,640], help="Overlaping stride of the tile [stride_x, stride_y]. Dafault: tile_size")
    parser.add_argument("--tile_width", type=int,  default=1280,  help="Width of a tile. Dafault: 1280")
    parser.add_argument("--tile_height", type=int, default=1280,  help="Heigth of a tile. Dafault: 1280")
    parser.add_argument("--multichannel", type=bool, default=True, help="Select if multichannel")
    parser.add_argument("--resolution", type=bool, default=False, help= "Read the resolution from .tiff files")
    parser.add_argument("--pad_borders",  type=bool, default=True, help="Pad borders to tile image")
    parser.add_argument("--pad_color", type=list, default=[255,255,255], help="Set pad borders RGB color. Default: [255,255,255] (white)")
    parser.add_argument("--log", type=str, default='NOTSET', help="Select logs level. Defaults NOTSET",
                        choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument('--classes', type=str, default="class_1, class_2", help="Select class labels separated by commas",)
    parser.add_argument("--balance_data", type=bool, default=True, help="Balance tiling object class")
    parser.add_argument("--label_format", type=str, default="yolo", help="Label Output format")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads to run")

    return parser.parse_args(), kwargs




if __name__==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="preprocessing", help="Key params")
    parser.add_argument("--params_file", default="params.yaml", help="Source file with the parameters")
    parser.add_argument("--log", type=str, default='NOTSET', help="Select logs level. Defaults NOTSET",
                        choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument("--workers", type=int, default=4, help="Number of Threads to run")
    args = parser.parse_args()

    # set up logger
    logger = set_logs(args.log)

    # Load the parameters
    with open(args.params_file, 'r') as fd:
        params = yaml.safe_load(fd)
        kwargs = params["preprocessing"]
    if args.params == "test_data":
        kwargs["images_path"] = params[args.params]["images_path"]
        kwargs["labels_path"] = params[args.params]["labels_path"]
        kwargs["save_path"]   = params[args.params]["save_path"]
    # Configure paths
    print(kwargs["images_path"])
    images_path = Path(kwargs["images_path"]) # Necessary argument
    labels_path = Path(kwargs.get("labels_path",images_path))

    if images_path.is_dir():
        # Get the preprocessing class object
        img_ext = set([e.suffix for e in images_path.rglob(f'*.*')  if e.suffix.lower() in IMAGE_EXT ])
        kwargs["ext"] = kwargs.get("ext", ','.join(img_ext)  )

        # Get the information of total file numbers
        n_files = len(list(images_path.rglob(f'*{ kwargs["ext"] }')))

        logging.info(f"Scanning for files in folder: {images_path} ... Total image files {n_files}")
        print(f"Processing {n_files} image files  in {images_path} ...  with {args.workers} workers")

        fnames = ((im_fname, labels_path/Path(im_fname.name).with_suffix(".txt") ) for im_fname in images_path.rglob(f'*{kwargs["ext"]}'))

    else:
        fnames = zip( load_text_file(images_path), load_text_file(labels_path))
    bbox_images_preprocessor = get_preprocessing_object(**kwargs)
    tile_bbox_dataset(fnames, bbox_images_preprocessor, args.workers)
