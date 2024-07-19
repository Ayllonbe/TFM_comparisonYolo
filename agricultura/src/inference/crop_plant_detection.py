import numpy
import os
from pathlib import Path
from .CropDetectionModels import CropDetection
from enum import Enum
import torch

def config_device(computing_device):
    if 'gpu' in computing_device:
        device_number = computing_device.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

class CropModels(Enum):
    """
    This class defines the types of algorithm implemented in the crop-plant-detection package.
    Args:
        Enum (_type_): Types of models
    """
    MULTICROP = 0    # Model trained with five crops dataset (BSNN, HELAN, ZEAMX, GLMXA, GOSHI, LACSA)
    BRSNN = 1       # Model trained with BRSNN (Cannola Oil Seed Rape)

class CropPlantDetection():
    def __init__(self, model_type:CropModels,  device='gpu:0', conf_threshold:float=None, iou_threshold:float=None, model_path:str=None) -> None:
        """
        This class creates a Crop Detection model.
        Args:
            model_type (CropModels): Model type from the models enumerated in CropModels
            iou_threshold (float, optional): NMS IoU threshold to be applied to the model. If none optimal value will be selected. Defaults to None.
            conf_threshold (float, optional): Confidence threshold to be applied to the model. If none optimal value will be selected. Defaults to None.
            device (str, optional): Device where the model will be loaded. Defaults to 'gpu'.
            model_path (str, optional): Optional path to (local) model file. If None, internal models will be loaded. Defaults to None.
        """
        self.iou = iou_threshold
        kwargs = {}
        if model_type == CropModels.BRSNN:
            kwargs['tile_size']=[1280,1280]
            kwargs['strides']=[640,640]
            if model_path is None or model_path=="":
                model_path = Path(__file__).parent/"models"/"BRSNN.pt"
    
        elif  model_type == CropModels.MULTICROP: 
            kwargs['tile_size']=[2048,2048]
            kwargs['strides']=[1024,1024]
            if model_path is None or model_path=="":
                model_path = Path(__file__).parent/"models"/"MULTICROP.pt"
            
        else:
            raise Exception("Model type has to be chosen from Crop Detection Models")
        #Check that the needed parameters are configured
        if not Path(model_path).exists():
                raise Exception(f"Model {model_path} was not found")
        if conf_threshold is not None: # confidence threshold (0-1)
            kwargs['conf_thres'] = float(conf_threshold)
        if iou_threshold is not None: # NMS IoU threshold (0-1)
            kwargs['iou_thresh'] = float(iou_threshold)

            #Configure device
        config_device(device)
        print(f"Configure GPU device to: {torch.cuda.device_count() - 1}")
        device = device.split(':')[-1] 

        model = CropDetection(model_path, device, kwargs)
        self.model = model


    def predict(self, image_rgb:numpy.ndarray, crops = []) -> tuple:
        """
        This function returns the output detections

        Args:
            image_rgb (numpy.ndarray): 0-255 range RGB image of dimensions [MxNx3]
        Returns:
            results: pd.Dataframe
                Results list data returned by yolov5 model after filtering with absolute coordinates of the objects
            result_image: PIL Image
                Result Image containing the bounding boxes of the detected crop.
        """
        return self.model.predict(image_rgb, crops)
