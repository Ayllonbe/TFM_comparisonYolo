from ultralytics import YOLO, YOLOWorld
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
      ####################################
      #       VARIABLE TO CHANGE         # 
      ####################################
      yoloworld = False
      name = "yolov5"
      modelname = 'yolov5nu.pt'
      ####################################
      ####################################

      if yoloworld:
            model = YOLOWorld(modelname)
      else:
      # Load a COCO-pretrained YOLOv8n model
            model = YOLO(modelname)
      # Display model information (optional)
      model.info()

      # Train the model on the COCO8 example dataset for 100 epochs
      results = model.train(data='./coco.yaml', epochs=100, name=name, exist_ok=True,imgsz=640, patience=20, device=0, batch=-1, cache=True, verbose=True, pretrained=False, workers=14)
      print("Training Finished")
