import json

def cargar_datos_json(fichero):
    with open(fichero, 'r') as archivo:
        datos = json.load(archivo)
    return datos
# Cargar datos JSON
data = cargar_datos_json("runs/detect/yoloworld/test/test_metrics.json")
# Nombres de las categorías
categories = data['categories']['names']

# Métricas
map50_95 = data['categories']['map50-95']
map50 = data['categories']['map50']
precision = data['categories']['precision']
recall = data['categories']['recall']
f1 = data['categories']['f1']

# Crear la tabla en formato LaTeX
latex_table = """
\\begin{longtable}[c]{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Categoría} & \\textbf{Precisión} & \\textbf{Recall} & \\textbf{map50-95} & \\textbf{map50} & \\textbf{F1} \\
    \\hline
    \\endfirsthead

    \\hline
    \\multicolumn{6}{|c|}{Continuación de la Tabla \\ref{tab:resultados3}}\\\\
    \\hline
    \\textbf{Categoría} & \\textbf{Precisión} & \\textbf{Recall} & \\textbf{map50-95} & \\textbf{map50} & \\textbf{F1} \\
    \\hline
    \\endhead

    \\hline
    \\endfoot

    \\endlastfoot
"""

# Añadir filas a la tabla
for i in range(len(categories)):
    category_name = categories[str(i)]
    latex_table += f"    {category_name} & {precision[i]:.3f} & {recall[i]:.3f} & {map50_95[i]:.3f} & {map50[i]:.3f} & {f1[i]:.3f} \\\\\n"

# Finalizar la tabla
latex_table += "\\caption{YOLOv9}\n\\label{tab:resultados3}\n\\end{longtable}"

# Imprimir la tabla LaTeX
print(latex_table)