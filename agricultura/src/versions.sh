#!/bin/bash
(
echo "Library Version"
echo "DVC $(dvc -V)"
echo "jisap-cli $(jisap-cli version)"
python3 -V
python3 -c "import yolov5; print(\"YoloV5\", yolov5.__version__)"
python3 -c "import torch; print(\"PyTorch\", torch.__version__)"
python3 -c "import torchvision; print(\"TorchVision\", torchvision.__version__)"
) | jisap-cli csv2md -s ' '