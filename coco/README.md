
# YOLO Object Detection with COCO Dataset

## Introduction

This repository contains a YOLO (You Only Look Once) implementation for object detection using the COCO dataset. YOLO is a state-of-the-art, real-time object detection system. This README provides an overview of the project, including setup instructions, how to run the model, and how to compare its performance with other models.

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/Ayllonbe/TFM_comparisonYolo.git
cd coco
pip install -r requirements.txt
```

Ensure you have CUDA installed if you plan to use a GPU for training.

## Dataset

The COCO dataset can be downloaded from the official [COCO website](https://cocodataset.org/#download). You will need the 2017 Train/Val dataset.

1. Download the COCO 2017 Train/Val dataset.
2. Extract the datasets into the `data/coco` directory:

```bash
mkdir -p data/coco
cd data/coco
# Extract the dataset here
```

## Training

To train the YOLO model on the COCO dataset, run the following command:

```bash
python yolo_train.py
```

This will start the training process. You can customize the configuration file and other parameters as needed.

This code will run one YOLO version. To add another version you must open the file and change it: yolov5nu.pt, yolov8n.pt, yolov9c.pt, yolov8-worldv2.pt.

## Evaluation

After training, you can evaluate the model's performance on the validation set:

```bash
python val.py 
```
It will run all the models used in this TFE at once.

This will output various metrics, including mAP (mean Average Precision), which is commonly used to evaluate object detection models.

For YOLOWorld, the line of code to run is:

```bash
python val_yoloworld.py 
```

To run inference using the trained YOLO model on new images, follow the steps below.
Running Inference on Images

    Place the images you want to run inference on in the data/images directory.
    Run the following command to perform inference:

bash
```
python inferencia.py  --image_file datasets/coco/test-dev2017.txt --weights detect\yolov5\weights\best.pt
```
This command will perform object detection on all images in the datasets/coco/test-dev2017.txt directory using the trained model weights. The results will be saved in the detect\yolov5 directory.

For yoloworld, the code to run is the following: 
bash
```
python inferencia_yoloworld.py  --image_file datasets/coco/test-dev2017.txt --weights detect\yoloworld\weights\best.pt
```




## Acknowledgements

- [YOLO](https://github.com/ultralytics/)
- [COCO Dataset](https://cocodataset.org/#home)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the sections according to your project's specifics, including adding more detailed instructions, results, or other relevant information.