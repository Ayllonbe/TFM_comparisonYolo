# YOLO Object Detection with BASF Agricultural Dataset

## Introduction

This repository contains a YOLO (You Only Look Once) implementation for object detection using the BASF Agricultural dataset. YOLO is a state-of-the-art, real-time object detection system. This README provides an overview of the project, including setup instructions, how to run the model, and how to compare its performance with other models.


## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/Ayllonbe/TFM_comparisonYolo.git
cd agricultura
pip install -r requirements.txt
```

Ensure you have CUDA installed if you plan to use a GPU for training.

## Dataset

The BASF Agricultural dataset needs to be downloaded and prepared for training.

1. Obtain the dataset from BASF.
2. Extract the datasets into the `dataset` directory:

```bash
mkdir -p dataset
cd dataset
# Extract the dataset here
```

3. Ensure the dataset is in the correct format (e.g., YOLO or COCO format).


## DVC (Data Version Control)

We use DVC (Data Version Control) to build and manage automatic pipelines for our YOLO object detection project on the BASF Agricultural dataset. DVC helps in tracking data files, machine learning models, and experiments efficiently.

### Setting Up DVC

First, ensure you have DVC installed:

```bash
pip install dvc
```

Initialize DVC in your project directory:

```bash
dvc init
```

### Creating Pipelines

DVC pipelines are defined in `dvc.yaml` files. Here’s an example of how you might set up your pipeline:


### Running the Pipeline

To execute the pipeline, use the `dvc repro` command. This command will run all stages of the pipeline in the correct order, checking for changes and re-running only the stages that need to be updated.

```bash
dvc repro
```

This will run the preprocessing, training, and evaluation stages in sequence, ensuring that all dependencies are up to date.

### Tracking Data and Models

DVC allows you to version control your data and models, making it easy to track changes and share your work. Use the following commands to track your data and model outputs:

### Benefits of Using DVC

- **Reproducibility**: Easily reproduce any stage of your pipeline with `dvc repro`.
- **Version Control**: Track changes in your data, code, and models.
- **Collaboration**: Share your work with others and collaborate efficiently by tracking data and models.


## Acknowledgements

- [YOLO](https://github.com/ultralytics)
- [DVC](https://dvc.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the sections according to your project's specifics, including adding more detailed instructions, results, or other relevant information.