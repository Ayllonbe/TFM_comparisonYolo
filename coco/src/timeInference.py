import pandas as pd
import matplotlib.pyplot as plt
import glob

# Ruta de los archivos CSV
df_yolov5 = pd.read_csv("inference_summary_time_yolov5.csv")
df_yolov8 = pd.read_csv("inference_summary_time_yolov8.csv")
df_yolov9 = pd.read_csv("inference_summary_time_yolov9.csv")
df_yoloworld = pd.read_csv("inference_summary_time_yoloworld.csv")

csv_files = {
    "YOLO v5 M" : df_yolov5,   
    "YOLO v8 M" : df_yolov8,   
    "YOLO v9 C" : df_yolov9,   
    "YOLOWorld M" : df_yoloworld
}
# Diccionario para almacenar los tiempos por modelo y etapa
data = {}

# Leer cada CSV y almacenar los tiempos en el diccionario
for model_name in csv_files.keys():
    df = csv_files[model_name]
    data[model_name] = {
        'preprocessing': df['preprocess'].tolist(),
        'inference': df['inference'].tolist(),
        'postprocessing': df['postprocess'].tolist()
    }

# Crear figuras para cada etapa
def create_boxplot(data, stage, title, xlabel):
    plt.figure(figsize=(10, 6))
    stage_data = [data[model][stage] for model in data]
    labels = [model for model in data]
    plt.boxplot(stage_data, labels=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Tiempo (ms)')
    plt.grid(True)
    plt.savefig(stage+"_boxplot.png")

# Diagramas de bigotes para cada etapa
create_boxplot(data, 'preprocessing', 'Tiempos de Preprocesamiento', 'Modelos')
create_boxplot(data, 'inference', 'Tiempos de Inferencia', 'Modelos')
create_boxplot(data, 'postprocessing', 'Tiempos de Postprocesamiento', 'Modelos')
