import os
import random

def list_random_images(folder, num_images=10):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            images.append(img_path)
    
    if len(images) < num_images:
        print(f"La carpeta contiene menos de {num_images} imágenes.")
        return images
    
    return random.sample(images, num_images)

def save_image_paths_to_file(image_paths, output_file):
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')

if __name__ == '__main__':
    folder_path = 'datasets/coco/images/test2017/'  # Reemplaza esto con la ruta de tu carpeta de imágenes
    output_file = 'imagenes_aleatorias.txt'  # Nombre del archivo de salida
    
    random_images = list_random_images(folder_path, 10)
    save_image_paths_to_file(random_images, output_file)
    
    print(f"Las rutas de las imágenes se han guardado en {output_file}")
