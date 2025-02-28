from pathlib import Path
import cv2
import argparse
import json
import numpy as np


def create_image_annotation(file_path: Path, width: int, height: int, image_id: int):

    file_path = file_path.name
    image_annotation = {
        "file_name": file_path,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation


def create_annotation_from_yolo_format(
        min_x, min_y, width, height, image_id, category_id, annotation_id, segmentation=True
):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    if segmentation:
        seg = [[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]
    else:
        seg = []
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": seg,
    }

    return annotation


# Create the annotations of the ECP dataset (Coco format)
coco_format = {"images": [{}], "categories": [], "annotations": [{}]}


classes = [
    "chair",
    "handle",
    "table",
    "button",
    "person",
]


def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg"))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for file_path in file_paths:
        # Check how many items have progressed
        if image_id % 1000 == 0:
            print("Processing " + str(image_id) + " ...")

        img_file = cv2.imread(str(file_path))
        h, w, _ = img_file.shape
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.txt"
        if opt.yolo_subdir:
            annotations_path = file_path.parent / opt.yolo_subdir / label_file_name
        else:
            annotations_path = file_path.parent / label_file_name

        if not annotations_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as f:
            yolo_labels = f.readlines()


        for lbl in yolo_labels:
            label_line = lbl
            category_id = (
                    int(label_line.split()[0]) + 1
            )  # you start with annotation id with '1'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            float_x_center = img_file.shape[1] * x_center
            float_y_center = img_file.shape[0] * y_center
            float_width = img_file.shape[1] * width
            float_height = img_file.shape[0] * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)

            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                image_id,
                category_id,
                annotation_id,
                segmentation=opt.box2seg,
            )
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images_annotations, annotations


def debug(opt):
    path = opt.path
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()

    # read the file
    file = open(path, "r")
    read_lines = file.readlines()
    file.close()

    for line in read_lines:
        print("Image Path : ", line)
        # read image file
        img_file = cv2.imread(line[:-1])

        # read .txt file
        label_path = line[:-4] + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()
        label_file.close()

        for line1 in label_read_line:
            label_line = line1

            category_id = label_line.split()[0]
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            int_x_center = int(img_file.shape[1] * x_center)
            int_y_center = int(img_file.shape[0] * y_center)
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)

            min_x = int_x_center - int_width / 2
            min_y = int_y_center - int_height / 2
            width = int(img_file.shape[1] * width)
            height = int(img_file.shape[0] * height)

            print("class name :", classes[int(category_id)])
            print("x_upper_left : ", min_x, "\t", "y_upper_left : ", min_y)
            print("width : ", width, "\t", "\t", "height : ", height)
            print()

            # Draw bounding box
            cv2.rectangle(
                img_file,
                (int(int_x_center - int_width / 2), int(int_y_center - int_height / 2)),
                (int(int_x_center + int_width / 2), int(int_y_center + int_height / 2)),
                color_list[int(category_id)],
                3,
            )

        cv2.imshow(line, img_file)
        delay = cv2.waitKeyEx()

        # If you press ESC, exit
        if delay == 27 or delay == 113:
            break

        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument("-p", "--path", type=str, help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images." )
    parser.add_argument(   "--debug",  action="store_true", help="Visualize bounding box and print annotation information" )
    parser.add_argument("--output", default="train_coco.json", type=str,  help="Name the output json file", )
    parser.add_argument(
        "--yolo-subdir",
        action="store_true",
        help="Annotations are stored in a subdir not side by side with images.",
    )
    parser.add_argument( "--box2seg",  action="store_true", help="Coco segmentation will be populated with a "
                                                                 "polygon that matches replicates the bounding box data.",
    )
    args = parser.parse_args()
    return args


def main(opt):
    output_name = opt.output
    output_path = "output/" + output_name

    print("Start!")

    if opt.debug is True:
        debug(opt)
        print("Debug Finished!")
    else:
        (
            coco_format["images"],
            coco_format["annotations"],
        ) = get_images_info_and_annotations(opt)

        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        with open(output_path, "w") as outfile:
            json.dump(coco_format, outfile, indent=4)

        print("Finished!")


if __name__ == "__main__":
    options = get_args()
    main(options)