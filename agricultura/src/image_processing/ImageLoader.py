from pathlib import Path
import argparse
import logging
import tifffile
import numpy as np
import cv2
import rawpy
from PIL import Image, ImageOps

try:
    import imageio.v2 as imageio
except:
    import imageio


class MinMax(object):

    def __init__(self,  min_val=30, max_val=220):
        self.min = min_val
        self.max = max_val
    def __call__(self, image):
        return np.clip(image,self.min,self.max)

def load_text_file(filename):
    """ Load a text binary file as a list of strings line by line

    Parameters
    ----------
    filename: str, Path
        File name of JSON file we want to load.

    Returns
    ----------
    data: Object
        Python object loaded be serialized into a JSON Object.
    """
    lines = []
    with open(filename, 'r') as f:
        # testList = f.readlines()
        for line in f:
            # remove linebreak
            lines.append( line[:-1])

    return lines


IMAGE_EXT =['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif', '.ppm', '.xbm', \
            '.tiff', '.rgb', '.pgm', '.png', '.pnm', '.cr2','.nef', '.x3f', '.kdc', '.ndpi', '.svs']



class ImageLoader(object):
    r""" Loads image from different sources and formats

    Attributes
    ----------
    image_shape: np.ndarray
        the size of the image window, the size of a single patch
    stride_size: np.ndarray
        the step size between patches in the input  spatial dimensions. The stride controls the overlap for the sliding blocks.
    pad_shape: np.ndarray
        Border padding controls the amount of color-padding on both sides for padding number of points for
                each spatial dimension.

    Math::
        Tiles_num = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d]  \times
        (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial\_size}` is formed by the spatial dimensions  of :attr:`input` (:math:`*` above),
    and :math:`d` is over all spatial  dimensions.
    Example Use:
    --------
        >>> image_path = 'example_image'
        >>> Loader= ImageLoader(tile_size= [2,2], tile_strides=[1,1], padding=True)
        >>> patches = ImageTiler(image)
    """

    IMAGE_EXT =['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif', '.ppm', '.xbm', \
            '.tiff', '.rgb', '.pgm', '.png', '.pnm', '.cr2','.nef', '.x3f', '.kdc', '.ndpi', '.svs']


    def __init__(self, ext = ".jpg",  padding=False, multichannel = False, save_dir = None, **kwargs):
        IMAGE_LOADERS ={ "raw_image": self.load_raw_image,
                         "multichannel_image": self.load_multichannel_image,
                         "microscopy_image": self.load_microscopy_image,
                         "image": self.load_microscopy_image,
                         "PIL_image": self.load_microscopy_image  }

        self.PADDING = padding
        self.pad = 0
        self.pad_color = kwargs.get("pad_color", [255,255,255] ) #RGB code (0-255) for padded pixels
        self.multichannel = multichannel
        self.ext = ext if "." in ext else "." + ext
        self.save_ext = kwargs.get("save_format", self.ext) # if not specified, same format as original
        self.full_image = None

        # Create directory in case of saving the images
        self.save_dir = Path(save_dir)
        if save_dir is not None:  Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        if ext.lower() in ['.cr2','.nef', '.x3f', '.kdc' ]:
            self.loader = self.load_raw_image
        if ext.lower() in ['.ndpi','.svs'] and multichannel:
            self.loader = self.load_microscopy_image
        elif multichannel:
            self.loader = self.load_multichannel_image
        else:
            self.loader = self.load_image

        # Select image writer if multichannel volume, else write 2D image
        self.writer = imageio.volwrite if (self.multichannel and self.save_ext.lower() in [".tiff", ".tif"]) else imageio.imwrite
        self.preprocessing_fn = []
        # self.preprocessing_fn += [ MinMax(min_val=0, max_val=255)]

    @staticmethod
    def load_image(image_path):
        image = Image.open(str(image_path)).convert('RGB')
        img_PIL = ImageOps.exif_transpose(image)
        img = np.array(img_PIL)
        return img # imageio.imread(image_path)
    
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
    def load_microscopy_image(image_path):
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

    @staticmethod
    def read_RGB_image_opencv(image_path):
        try:
            srcBGR = cv2.imread(image_path,  cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
            return destRGB
        except:
            return None
    @staticmethod
    def save_image( image, path):
        if len(image.shape) == 3 and image.shape[-1] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img = image
        return cv2.imwrite(path, img)#, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def pad_size(self):
        return self.pad[::-1]

    def __call__(self, image_path, *args, **kwargs):
        # Read the multichannel image
        image = self.loader(image_path)
        image = [f(image) for f in self.preprocessing_fn][0] if self.preprocessing_fn != [] else image

        if self.PADDING: #pad the borders
            offset = self.__offset(image.shape[-3:-1])
            self.pad =  np.array(image.shape[-3:-1]) - offset if  offset.all() > 0 else np.zeros_like( image.shape[-3:-1])
            image = self.__pad_borders(image, self.pad , color = self.pad_color )

        # Return the list generators
        return image


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

    def __offset(self, image_size, pad_div):
        """  Auxiliary methods to compute the offset for make exactly divisible"""
        image_size = np.array(image_size)
        offset =  np.array(image_size) - (pad_div * np.floor(np.array(image_size) / pad_div))
        return offset

    def save_RGB_image_opencv(self, image, filename,folder='images', **kwargs):
        image_path = self.save_dir/folder / filename.replace(f'{self.ext}', f'{self.save_ext}')
        image_path.parent.mkdir(exist_ok=True, parents=True)
        self.save_image( image, str(image_path))
        # self.writer(str(image_path), image, **kwargs)

    @staticmethod
    def resize_image(image,  size,  interpolation = cv2.INTER_LANCZOS4):
        return cv2.resize(image, size, interpolation = interpolation)

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
            image = tifffile.TiffFile(str(image_path))
            try: # Try to extract resolution from Tiff image
                resolution[0] = image.pages[0].tags['XResolution'].value[0] / image.pages[0].tags['XResolution'].value[1]
                resolution[1] = image.pages[0].tags['YResolution'].value[0] / image.pages[0].tags['YResolution'].value[1]
            except:
                logging.error(  f"{str(image_path.name)}: meta-data could not be read or resolution info is not available")
        return resolution

    @staticmethod
    def extension(ext):
        return '*.' + ''.join('[%s%s]' % (e.lower(), e.upper()) for e in ext)

def args_parser(**kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", default="dataset",
                        help="Source folder with images  to be tiled")
    parser.add_argument("--target_path", default="resized_dataset",   help="Target folder for a new sliced dataset")
    parser.add_argument("--ext", default="JPG", help="Image extension in a dataset. Default: .JPG")
    parser.add_argument("--save_format", default=".JPG", help="Image extension in a dataset. Default: .png")
    parser.add_argument("--multichannel", type=bool, default=False, help="Select if multichannel")
    parser.add_argument("--resolution", type=bool, default=False, help= "Read the resolution from .tiff files")
    parser.add_argument("--pad_borders",  type=bool, default=True, help="Pad borders to tile image")
    parser.add_argument("--pad_color", type=list, default=[0,0,0], help="Set pad borders RGB color. Default: [0,0,0] (black)")
    parser.add_argument("--log", type=str, default='NOTSET', help="Select logs level. Defaults NOTSET",
                        choices=['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument("--workers", type=int, default=4, help="Number of threads to run")
    parser.add_argument("--max_size", type=int, default=5000, help="Number of threads to run")

    return parser.parse_args(), kwargs


def resize_files(fnames, Loader, max_size):

    # Iterate over all the files in the directory that has the corresponding label
    for fname in fnames:
        im_fname = Path(fname)
        image = Loader(im_fname)
        image_dim = np.array(image.shape[-3:-1])[::-1]
        if (image_dim > args.max_size).any():
            scale = np.max(np.round(image_dim / max_size))
            logging.info(f"RESIZING {im_fname.name} TO SCALE: {scale}")
            resize_image = Loader.resize_image(image, tuple(np.array(image_dim // scale).astype(int)))
            Loader.save_RGB_image_opencv(resize_image, im_fname.name, folder= im_fname.relative_to(images_path).parent)
        else:
            Loader.save_RGB_image_opencv(image, im_fname.name, folder= im_fname.relative_to(images_path).parent)


def load_fnames(images_path, args):

    if images_path.is_file():

        fnames = load_text_file(images_path)
        img_ext = list(set([Path(e).suffix for e in fnames if Path(e).suffix.lower() in IMAGE_EXT]))[0]

    elif len(list(images_path.rglob('images'))) != 0:

        fnames = Path(images_path).rglob(f'*/images/{ImageLoader.extension(args.ext)}')
        img_ext = args.ext
    else:
        try:
            img_ext = list(set([e.suffix for e in images_path.rglob(f'*.*') if e.suffix.lower() in IMAGE_EXT]))[0]
            fnames = images_path.rglob(f'*{img_ext}')
        except:
            fnames = images_path.rglob(f'.*')
            img_ext = list(images_path.rglob(f'.*'))[0].suffix
    return fnames, img_ext

if __name__==  '__main__':

    args, kwargs = args_parser()

    # Configure paths
    images_path = Path(args.images_path)
    target_path = Path(args.target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    fnames, img_ext = load_fnames(images_path, args)
    # Create loader Object
    Loader = ImageLoader(ext = img_ext, save_dir=args.target_path)
    # Resize the images if size above max_size
    resize_files(fnames, Loader, args.max_size)
