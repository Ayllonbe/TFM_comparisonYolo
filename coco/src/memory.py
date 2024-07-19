import torch
import torch.cuda
from ultralytics import YOLO

# Cargar un modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt').model

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Crear una entrada dummy con el tamaño de imagen que espera el modelo
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# Limpiar la memoria caché antes de la inferencia
torch.cuda.empty_cache()

# Monitorear el uso de memoria antes de la inferencia
memory_before = torch.cuda.memory_allocated(device)

# Realizar la inferencia
with torch.no_grad():
    output = model("coco.yaml")

# Monitorear el uso de memoria después de la inferencia
memory_after = torch.cuda.memory_allocated(device)

# Calcular el uso de memoria durante la inferencia
memory_used = memory_after - memory_before
print(f'Memoria utilizada durante la inferencia: {memory_used / 1024**2:.2f} MB')
