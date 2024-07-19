from pathlib import Path
import argparse
import tifffile
import imageio
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from more_itertools import ichunked, chunked

try:
    from src.image_processing.ImageTransforms import *
    from src.utils.logs import set_logs
except ModuleNotFoundError:
    import sys
    sys.path.append('src')
    sys.path.append('src/image_processing')
    sys.path.append('rc/utils')
    from image_processing.ImageTransforms import *
    from utils.logs import set_logs
try:
    import rawpy
except: 
    print("Raw image mode could not be imported")


logger = set_logs('DEBUG', name='tiler.log')

IMAGE_EXT =['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif', '.ppm', '.xbm', \
            '.tiff', '.rgb', '.pgm', '.png', '.pnm', '.cr2','.nef', '.x3f', '.kdc', '.ndpi', '.svs']


class ImageTiler(object):
    def __init__(self, tile_size, tile_strides, padding=True, multichannel = False, ext = ".jpg", save_dir = None, **kwargs):
        self.tile_shape = np.array(tile_size) if len(tile_size) >= 2 else  np.array([tile_size, tile_size])
        self.tile_stride = np.array(tile_strides)[::-1] if len(tile_strides)>=2 else np.array([tile_strides, tile_strides]) # Flip coordinates to match numpy convention
        self.PADDING = padding
        self.pad = 0
        self.pad_color = kwargs.get("pad_color", [255,255,255] ) #RGB code (0-255) for padded pixels
        self.multichannel = multichannel
        self.ext = ext if "." in ext else "." + ext
        self.save_ext = kwargs.get("save_format", self.ext) # if not specified, same format as original
        self.full_image = None

        # Create directory in case of saving the images
        self.save_dir = None
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            Path(self.save_dir /"images").mkdir(parents=True, exist_ok=True)

        if ext.lower() in ['.cr2','.nef', '.x3f', '.kdc' ]:
            self.loader = self.load_raw_image
        if ext.lower() in ['.ndpi','.svs'] and multichannel:
            self.loader = self.load_microscopy_image_multilayer
        elif multichannel:
            self.loader = self.load_multichannel_image
        else:
            self.loader = self.load_image

        # Select image writer if multichannel volume, else write 2D image
        self.writer = imageio.volwrite if (self.multichannel and self.save_ext.lower() in [".tiff", ".tif"]) else imageio.imwrite
        self.preprocessing_fn = []
        self.preprocessing_fn += [ MinMax(min_val=0, max_val=255)]

    def __tile(self, image, tile_x, tile_y):
        # Tile Image generator for the given full image
        idx =+ tile_x * self.tile_stride[0]
        idy =+ tile_y * self.tile_stride[1]
        yield image[..., idx:idx + self.tile_shape[0], idy:idy+self.tile_shape[1],  :]


    def __tile_from_xy(self, image, x_center, y_center):
        # Tile image generator for the given (x,y)
        idx = max(int(x_center) - (self.tile_shape[0]) // 2, 0)
        idy = max(int(y_center) - (self.tile_shape[1]) // 2, 0)
        yield image[..., idx:idx + self.tile_shape[0], idy:idy+self.tile_shape[1], :]



    @staticmethod
    def load_image(image_path):
        return imageio.imread(image_path)

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
        image = tifffile.imread(image_path)
        return np.array(image)

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



    def tile_generator(self, image_path):

        self.full_image = self.loader(image_path)

        # Preprocesss the image (optional)
        if len(self.preprocessing_fn)>0:
            self.full_image = [f(self.full_image) for f in self.preprocessing_fn][0]
        image = self.full_image

        # Tile image dimensions (numpy flips height and width shapes)
        height, width  = image.shape[-3:-1]

        if self.PADDING: #pad the borders
            # offset = self.__offset(image.shape[-3:-1])
            offset = self.__offset( (   height, width))

            logging.debug(f"PADDING OFFSET  {offset} for image {image.shape[-3:-1]}")
            # self.pad =  1.5 * self.tile_stride - offset if  offset.all() > 0 else np.zeros_like( self.tile_stride)
            self.pad =  (self.tile_stride - offset ) + (self.tile_shape- self.tile_stride)

            logging.debug(f"PADDING BORDERS with {self.pad}")
            image = self.__pad_borders(image, self.pad.astype(int) , color = self.pad_color, center=True )

        height, width  = image.shape[-3:-1]

        n_columns   = np.array(width) // self.tile_stride[1] if  self.tile_stride[1] == self.tile_shape[1] else np.array(width) // self.tile_stride[1] - 1
        n_rows      = np.array(height) // self.tile_stride[0] if  self.tile_stride[0] == self.tile_shape[0] else np.array(height) // self.tile_stride[0] - 1

        #Create the generator for storing the tile images
        tile_images = (next(self.__tile(image, r, c))  for r in range(n_rows) for c in range(n_columns)  )

        return tile_images, ( n_rows, n_columns)

    def pad_size(self):
        return self.pad

    def stride_size(self):
        return self.tile_stride

    def tile_size(self):
        return self.tile_shape


    @staticmethod
    def __pad_borders(image, pad, color = [0, 0, 0], center = True):
        im_size = tuple( image.shape[-3:-1] + pad)
        padded_shape = image.shape[:-3] + tuple( im_size) + (image.shape[-1], )
        # Create padded image as filled with color (RGB) values TODO: optimize padded_image creation
        padded_image = np.full(shape=padded_shape, fill_value = color, dtype=image.dtype)

        if center:
            padded_image[...,pad[0]//2:(pad[0]//2)+image.shape[-3],  pad[1] // 2:( pad[1] // 2) +image.shape[-2], :] = image
        else: # Pad only on right bottom boarder not to modify bounding box coordinates
            padded_image[...,:image.shape[-3], :image.shape[-2], :] = image
        return padded_image

    def __offset(self, image_size):
        """  Auxiliary methods to compute the offset for make tilling exactly divisible"""
        # image_size = np.array(image_size)
        offset = (np.array(image_size)) - (self.tile_stride * ((image_size - self.tile_shape) // self.tile_stride) + self.tile_shape)
        offset =  np.array(image_size) - (self.tile_stride * np.floor(np.array(image_size) / self.tile_stride) )

        return offset

    def save_image( self, tile_image, filename, **kwargs):
        slice_path = self.save_dir /"images"/ filename.replace(f'{self.ext}', f'{self.save_ext}')
        self.writer(str(slice_path), tile_image, **kwargs)


    @staticmethod
    def __new_image_size( image_size,  stride, offset):
        """
        Compute new image size
        """
        return image_size - offset + stride + 1


    def __call__(self, image_path, *args, **kwargs):

        self.full_image = self.loader(image_path)

        # Preprocesss the image (optional)
        if len(self.preprocessing_fn)>0:
            self.full_image = [f(self.full_image) for f in self.preprocessing_fn][0]
        image = self.full_image

        # Tile image dimensions (numpy flips height and width shapes)
        height, width  = image.shape[-3:-1]

        if self.PADDING: #pad the borders
            # offset = self.__offset(image.shape[-3:-1])
            offset = self.__offset( (   height, width))

            logging.debug(f"PADDING OFFSET  {offset} for image {image.shape[-3:-1]}")
            # self.pad =  1.5 * self.tile_stride - offset if  offset.all() > 0 else np.zeros_like( self.tile_stride)
            self.pad =  (self.tile_stride - offset ) + (self.tile_shape- self.tile_stride)

            logging.debug(f"PADDING BORDERS with {self.pad}")
            image = self.__pad_borders(image, self.pad.astype(int) , color = self.pad_color, center=True )

        height, width  = image.shape[-3:-1]

        n_columns   = np.array(width) // self.tile_stride[1] if  self.tile_stride[1] == self.tile_shape[1] else np.array(width) // self.tile_stride[1] - 1
        n_rows      = np.array(height) // self.tile_stride[0] if  self.tile_stride[0] == self.tile_shape[0] else np.array(height) // self.tile_stride[0] - 1

        #Create the generator for storing the tile images
        tile_images = (next(self.__tile(image, r, c))  for r in range(n_rows) for c in range(n_columns)  )

        return tile_images, ( n_rows, n_columns)


def args_parser(**kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", default="../reviewed_labels", help="Source folder with images  to be tiled")
    parser.add_argument("--target_path", default="../tile_dataset_balanced/",   help="Target folder for a new sliced dataset")
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
    parser.add_argument('--classes', type=str, default="Binucleus,Micronucleus,Mononucleus,Multinucleus")
                        #default="class_1, class_2, class_3", help="Select class labesl separated by commalts NOTSET",)
    parser.add_argument("--balance_data", type=bool, default=False, help="Balance tiling object class")
    parser.add_argument("--label_format", type=str, default="yolo", help="Label Output format")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads to run")

    return parser.parse_args(), kwargs




def conf_image_tiler(params: dict, **kwargs):
    # Configure paths
    images_path = Path(params.get("images_path", kwargs.get("images_path", ) ) )
    # Get the image extension if not specified in params file
    params["ext"] = params.get("ext", ','.join(set([e.suffix for e in images_path.rglob(f'*.*')  if e.suffix in IMAGE_EXT ]))  )
    return get_image_tiler(**params)


def get_image_tiler(**kwargs):

    # Configure Tiling image options or default values
    tile_size   = kwargs.get("tile_size", [1280,1280])
    tile_stride = kwargs.get("tile_stride", tile_size)

    # Remove ambiguous values
    kwargs.pop("tile_size", None)
    kwargs.pop("tile_stride", None)
    # Create image tiler object
    return ImageTiler(tile_size=tile_size, tile_strides=tile_stride, **kwargs)



if __name__==  '__main__':
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml", help="Source file with the parameters")
    parser.add_argument("--log", type=str, default='DEBUG', help="Select logs level. Defaults NOTSET",
                        choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument("--workers", type=int, default=1, help="Number of Threads to run")
    args = parser.parse_args()

    # set up logger
    logger = set_logs(args.log)

    # Load the parameters
    with open(args.params, 'r') as fd:
        kwargs = yaml.safe_load(fd)['inference_data']

    # Configure paths
    images_path = Path(kwargs["images_path"]) # Necessary argument

    # Get the preprocessing class object
    img_ext = set([e.suffix for e in images_path.rglob(f'*.*') if e.suffix.lower() in IMAGE_EXT ])
    kwargs["ext"] = kwargs.get("ext", ','.join(img_ext)  )

    image_tiler = get_image_tiler(**kwargs)

    # Get the information of total file numbers
    n_files = len(list(images_path.rglob(f'*{ kwargs["ext"] }')))

    logging.info(f"Scanning for files in folder: {images_path} ... Total image files {n_files}")
    print(f"Total image files {n_files} in {images_path} ... processing with {args.workers} workers")

    fnames = images_path.rglob(f'*{kwargs["ext"]}')
    target_path = Path("tile_data")
    target_path.mkdir(parents=True, exist_ok=True)

    if args.workers > 1: # Load the data with multiple threads
        logging.info(f"Number of workers  {args.workers}")
        # Use threads to speed up I/O operations
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(image_tiler, fnames), desc="Tilling Images"))  # pool.map(lambda p: tiler(*p), fnames)
    else:
        # Iterate over all the files in the directory that has the corresponding label
        for im_fname in tqdm(fnames,  desc="Tilling Images", total = n_files ):
            tile_images,  n_tiles = image_tiler(im_fname)
            for r, img_row in enumerate(ichunked(tile_images, n_tiles[-1])):
                for i, tile in enumerate(img_row):
                    img = Image.fromarray(tile).convert("RGBA")
                    print(f"{im_fname.stem}-tile-{ str(r).zfill(2)}-{str(i).zfill(2)}")
                    img.save(target_path/Path(f"{im_fname.stem}-tile-{ str(r).zfill(2)}-{str(i).zfill(2)}").with_suffix(".png"))
