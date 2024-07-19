import numpy
import numpy as np
import cv2
import mahotas as mh


class ImagePreprocessing(object):
    def __init__(self, **kwargs):
        self.callbacks = None


class GammaCorrection(object):
    """Gamma correction nonlinear operation used to encode and decode luminance
    P_{\gamma} = P_{max}\cdot ( \frac{P_{img}}{P_{max}})^{1/\gamma}
    :param:         output_size (tuple or int): Desired output size. If int, square crop is made.
    """
    def __init__(self, gamma=0.55, px_max = 255.0):
        assert isinstance(gamma, float)
        assert isinstance(gamma, (int,float))
        self.gamma = gamma
        self.px_max =  px_max
        # build a gamma_ranges lookup table mapping the pixel values to gamma values
        self.gamma_mapping = np.array([( (p / px_max) ** (1.0 / gamma)) * px_max for p in np.arange(0, int(px_max)+1)]).astype("uint8")

    def __call__(self, image):
        # apply gamma correction using the lookup table
        return cv2.LUT(image, self.gamma_mapping )


class MinMax(object):

    def __init__(self,  min_val=30, max_val=220):
        self.min = min_val
        self.max = max_val
    def __call__(self, image):
        return np.clip(image,self.min,self.max)

class MergeChannels(object):

    def __init__(self,  approach="None", axis_ch=0):
        operations = {"min": np.min, "mean":np.mean, "median": np.median2, "std":np.std}
        self.merge = operations.get(approach, np.mean)
        self.axis = axis_ch

    def __call__(self, image):
        return self.merge(image,self.axis)

    @staticmethod
    def select_channel(image, n_channel=0):
        rgb_img = np.array(image)[n_channel,...]
        return rgb_img

class SharpestLayer(object):

    def __init__(self, n_layer=1, approach="sharpest"):
        sharpest = {"focus":  self.focus, "sharpest_canny": self.select_sharpest_canny,
                    "sharpest": self.determine_sharpest_layer}
        self.n_layer = n_layer
        self.select_sharpest = sharpest.get(approach, self.determine_sharpest_layer)


    def __call__(self, image):
        return self.select_sharpest(image)

    @staticmethod
    def focus(image, **kwargs):
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

    @staticmethod
    def determine_sharpest_layer(image, **kwargs):
        """Determine sharpest layer by that a less sharp image blurred,
        will have less  pixel-values difference thatn a blurred sharp images """
        kernel_size = kwargs.get("kernel_size", (5, 5))
        blur_diff = []
        for layer_image in image:
            blur_layer = cv2.blur(np.array(layer_image), kernel_size)
            blur_diff.append( np.sum(np.abs( blur_layer - layer_image)))
        # Select the sharpest image as index of the sum
        idx = np.argmax(blur_diff) #diff_sums.index(np.max(diff_sums))
        return image[idx]

    @staticmethod
    def get_canny_thr(image, percent=0.33):
        """ Get automatic thresholds ncv2"""
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


    def select_sharpest_canny(self, image, **kwargs):
        # Convert images to rgb
        images_gray = [cv2.cvtColor(img_layer, cv2.COLOR_BGR2GRAY) for img_layer in image]

        # Compute Canny values thresholds Automatically
        trh_1, trh_2 = self.get_canny_thr(images_gray, percent=0.33)

        # Get the shorted indices of sharpest layers
        edge_sharpness = [self.get_edge_sharpness(img_layer, trh_1, trh_2) for img_layer in images_gray ]

        idx_sharp = np.argsort(edge_sharpness)[-self.n_layer:]

        # Select the sharpest n_layers
        image = np.array(image)[idx_sharp,...]

        return numpy.squeeze(image)
