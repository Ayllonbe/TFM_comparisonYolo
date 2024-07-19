
from numpy.lib.stride_tricks import as_strided
import numpy as np




class ImagePatcher(object):
    r""" Extracts sliding local blocks from a np.ndarray image (grayscale, RGB or 3D stack image)
      into small patches given the tile size and the overlapping stride size.

    Attributes
    ----------
    tile_shape: np.ndarray
        the size of the sliding window, the size of a single patch
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
        >>> image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        >>> ImageTiler= ImagePatcher(tile_size= [2,2], tile_strides=[1,1], padding=True)
        >>> patches = ImageTiler(image)
    """

    def __init__(self, tile_size, tile_strides, padding=True,  **kwargs):
        self.tile_size = tile_size
        self.stride_size = tile_strides
        self.PADDING = padding
        self.pad_color = kwargs.get("pad_color", [255,255,255] ) #RGB code (0-255) for padded pixels
        self.preprocessing_fn = list(kwargs.get('preprocessing_fn', []))
        self.full_image = None




    @property
    def pad_shape(self):
        return self._pad[::-1]

    @pad_shape.setter
    def pad_shape(self, val):
        self._pad = val

    @property
    def stride_size(self):
        return self._tile_stride

    @stride_size.setter
    def stride_size(self, tile_strides):
        if len(tile_strides) >= 2:
            # Flip coordinates to match numpy convention
            self._tile_stride = np.array(tile_strides)[::-1]
        else:
            self._tile_stride = np.array([tile_strides, tile_strides])

    @property
    def tile_size(self):
        return self._tile_shape

    @tile_size.setter
    def tile_size(self, tile_size):
        # Ensure the tile shape is stored at least at 2 dim array
        if len(tile_size) >= 2:
            self._tile_shape =  np.array(tile_size)
        else:
            self._tile_shape =  np.array([tile_size, tile_size])

    def __tile(self, image, tile_x, tile_y):
        """  Tile Image for the given full image by specific tile idx"""
        self.idx =+ tile_x * self._tile_stride[0]
        self.idy =+ tile_y * self._tile_stride[1]
        return image[..., self.idx:self.idx+self._tile_shape[0], self.idy:self.idy + self._tile_shape[1], :]

    def __offset(self, image_size):
        """  Auxiliary methods to compute the offset for make tilling exactly divisible"""
        image_size = np.array(image_size)
        offset = (image_size) - (self._tile_stride * ((image_size - self._tile_shape) // self._tile_stride) + self._tile_shape)
        offset =  np.array(image_size) - (self._tile_stride * np.floor(np.array(image_size) / self._tile_stride) )

        return offset

    @staticmethod
    def pad_borders(image: np.ndarray, pad: np.ndarray, color = [0, 0, 0], center = True):
        color = color if image.shape[-1] == len(color) else color[:image.shape[-1]]
        im_size = tuple( image.shape[-3:-1] + pad)
        padded_shape = image.shape[:-3] + tuple( im_size) + (image.shape[-1], )
        # Create padded image as filled with color (RGB) values
        padded_image = np.full(shape=padded_shape, fill_value = color, dtype=image.dtype)
        if center:
            padded_image[...,pad[0]//2:(pad[0]//2)+image.shape[-3],  pad[1] // 2:( pad[1] // 2) +image.shape[-2], :] = image
        else: # Pad only on right bottom boarder not to modify bounding box coordinates
            padded_image[...,:image.shape[-3], :image.shape[-2], :] = image
        return padded_image

    @staticmethod
    def __new_image_size( image_size,  stride, offset):
        """  Compute new image size """
        return image_size - offset + stride + 1

    def __call__(self, image: np.ndarray, *args, **kwargs)->np.ndarray:

        if len(image.shape)==2:
            # If the image is grayscale, expand the array to have 3 dimensions (H,W,C=1)
            image = image[..., np.newaxis]
        self.full_image= image.copy()

        if self.PADDING: #Pad the borders
            self.pad_shape =  (self._tile_stride -  self.__offset( ( image.shape[-3:-1])) ) + (self._tile_shape- self._tile_stride)
            image = self.pad_borders(image, self._pad.astype(int) , color = self.pad_color, center=True )

        tile_images = self.tile_full_image(image)

        # Returns array with (I_r,I_c,H,W,C)
        return tile_images

    def tile_full_image(self, image):
        # Store needed image shape
        im_shape = np.array(image.shape)
        # Add channel dimensions to shape in case of RGB
        window_shape = np.array(tuple(self._tile_shape) + (im_shape[-1],), dtype=im_shape.dtype)
        # Compute slices indexes
        overlap_stride = tuple(self._tile_stride) + (im_shape[-1],)
        slices = tuple(slice(None, None, st) for st in overlap_stride)
        # Compute the shape of the new array as sequence of int
        tiles_num = ((im_shape - window_shape) // np.array(overlap_stride)) + 1
        new_shape = tuple(list(tiles_num) + list(window_shape))
        # Compute the strides of the new array
        strides = tuple(list(image[slices].strides) + list(np.array(image.strides)))
        # Create a view into the array with the given shape and strides.
        tile_images = np.squeeze(as_strided(image, shape=new_shape, strides=strides))
        return tile_images
