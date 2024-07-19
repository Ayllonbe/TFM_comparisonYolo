import logging
import pandas as pd
from yolov5 import hubconf
from yolov5.models.common import Detections
from PIL import Image
try:
    from image_processing.ImagePatcher import *
    from infer import plot_full_image_bbox, in_center_roi

except Exception as e:
    import sys
    from pathlib import Path
    sys.path.insert(0, Path(__file__).parent.parent)
    sys.path.append(Path(__file__).parent.parent.as_posix())
    from image_processing.ImagePatcher import *
    from infer import plot_full_image_bbox, in_center_roi


CROPS = ["BRSNN, HELAN, GLMXA, ZEAMX, GOSHI, LACSA"]

class CropDetection(object):
    r""" Class that encloses the Tiled crop detection yolov5 algorithm .

       Attributes
       ----------
       model_path: str
       pad_shape: np.ndarray
           Border padding controls the amount of color-padding on both sides for padding number of points for
                   each spatial dimension.
       kwargs: dict

       --------
           # >>> Model= CropDetection( model_path, device='cpu')
       """

    def __init__(self, model_path:str='models/MULTICROP.pt', device:str='gpu', kwargs:dict={}) -> None:
        self.model_path = model_path
        self.device = device
        self.model = self.load_yolo_model(self.model_path, device = device)
        tile_shape = kwargs.get('tile_size', [2048,2048])
        tile_strides =  kwargs.get('strides', [1024,1024])
        self.ImageTiler = ImagePatcher( tile_size=tile_shape, tile_strides=tile_strides, padding=True)
        
        # Modify the model configurations thresholds for prediction
        self.model.conf = float(kwargs.get("conf_thres",  self.model.conf))   # confidence threshold (0-1)
        self.model.iou  = float(kwargs.get("iou_thresh",  self.model.iou))    # NMS IoU threshold (0-1)
        self.model.agnostic=kwargs.get("nms-agnostic",  True)                 # Option to make the NMS post processing classs agnostic (bool)

    def predict(self, image: np.ndarray, crops: list = [ ]):
        """
        Helper function to predict Yolov5 model inference. If the image size is bigger that the tile shape defined
        the sliding window approach over the image is computed. See
        Parameters
        ----------
            image: PIL.Image

        Returns
        -------
            results: pd.Dataframe
            result_image: PIL.Image
        """
        tile_shape =  self.ImageTiler.tile_size

        results = []
        image_tiles =  self.ImageTiler(np.array(image)).astype(np.uint8)
        for b, img_row in enumerate(image_tiles):
            # Batch Inference by row
            preds = self.model(list(img_row), size=tile_shape)# [0])  # batched inference including NMS
            # Filter models that are on the overlap-stride region
            preds_df = filter_results_yolov5(preds, tile_shape,
                                                stride_size=self.ImageTiler.stride_size,
                                                pad=self.ImageTiler.pad_shape, batch=b)
            results.append(preds_df)
        # Return the results as pandas Dataframe
        results_df = pd.concat(results)
        if crops:
            # Filter the results to match only th especified  crop name
            results_df = results_df[results_df['name'].isin(crops)]
        result_image = plot_full_image_bbox(self.ImageTiler.full_image, results_df )

        return results_df, result_image

    @staticmethod
    def load_yolo_model(weights_path, device='cpu', autoshape=True):
        """ Load the yolov5 Hubconf model with qutoshape that includes NMS.  It allows inference from various sources.

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
        logging.debug("Loading Model: ",weights_path)
        model = hubconf.custom(path=weights_path, autoshape=autoshape, device=device)
        model.eval()
        return model


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

                preds_df.append([max(x1,0), max(y1,0), x2, y2, conf, cls, name])

                p += 1

    return pd.DataFrame(preds_df, index=indx, columns=results_df[0].columns)

if __name__=='__main__':
    
    import numpy as np
    model = CropDetection(model_path="/data2/projects/Comparison_object_detection/crop_counting_yolo_ultralytics/runs/detect/yolov5/train/weights/best.pt", device='cpu')
    file = "test_image_lacsa.JPG"
    img = np.array(Image.open(file))
    results, image = model.predict(img)
    image.save("output_image.jpg", quality=10)
