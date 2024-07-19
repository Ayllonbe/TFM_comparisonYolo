import os
import argparse
from ultralytics import YOLOWorld
from PIL import Image

def load_images_from_file(file_path):
    with open(file_path, 'r') as file:
        images = [line.strip() for line in file.readlines() if line.strip()]
    return images

def infer_and_save(model, images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"El archivo {img_path} no existe.")
            continue
        
        img = Image.open(img_path)
        results = model(img)
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename=output_path)  # save to disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", required=True, help="File containing image paths for inference")
    parser.add_argument("--model_path", required=True, help="Path to the YOLO model")
    parser.add_argument("--output_folder", default="output", help="Folder to save the output images")
    args = parser.parse_args()
    
    images = load_images_from_file(args.image_file)
    model = YOLOWorld(args.model_path)
    
    infer_and_save(model, images, args.output_folder)
