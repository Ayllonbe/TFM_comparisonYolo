from ultralytics import YOLO, YOLOWorld
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

modelList = ['yolov5nu.pt','yolov5su.pt','yolov5mu.pt','yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov9c.pt','yolov8s-worldv2.pt', 'yolov8m-worldv2.pt']

for m in modelList:
    if "world" in m:
        model = YOLOWorld(m)
    else:
        # Cargar un modelo YOLOv8 preentrenado
        model = YOLO(m)  # Puedes cambiar 'yolov8n.pt' por cualquier otro modelo disponible
    print(m)
    # Obtener el número total de parámetros
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f'Número total de parámetros: {total_params}')

    # Obtener el número de parámetros entrenables
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f'Número de parámetros entrenables: {trainable_params}')
    # Crear una entrada dummy con el tamaño de imagen que espera el modelo
    dummy_input = torch.randn(1, 3, 640, 640).to(model.device)

    # Pasar el modelo a evaluación
    model.model.eval()

    # Calcular los FLOPs
    flops = FlopCountAnalysis(model.model, dummy_input)
    print(f'FLOPs: {flops.total()}')

