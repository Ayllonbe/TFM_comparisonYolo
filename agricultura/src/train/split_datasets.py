from pathlib import Path
import shutil
import argparse
import numpy as np
from collections import Counter
import re
from re import match
import os
import sys
import logging

curr_path = os.path.dirname(os.path.realpath(__file__))
curr_path = curr_path.replace('\\','/')
main_path,_ = curr_path.rsplit('/',1)
main_path,_ = main_path.rsplit('/',1)
sys.path.append(main_path)
from src.utils.utils_io import load_text_file, save_text_file


def read_annotations_yolo(filename):
    """read annotation from annotation text file (.txt) saved as yolo """
    # Check in case of error if is in the same dir than the annotation
    label_file = Path(filename) if Path(filename).exists() else Path(filename).with_suffix(".txt")
    bboxes = []
    with open(label_file, 'r') as fin:
        for line in fin:
            label = line.rstrip('\n').split(' ')
            label_id = label[0]
            x_center =  float(label[1])
            y_center =  float(label[2])
            width =     float(label[3])
            heigth =    float(label[4])
            bboxes.append( [label_id, x_center,y_center,width,heigth,] )
    return bboxes

def get_all_class_labels(labels_dir):
    # Count all class indexes
    bbox_labels = []
    for val_label in Path(labels_dir).glob(f'*.txt') :
        yolo_bbox = read_annotations_yolo(val_label)
        # Select the label from annotation and flatten the lists
        labels = [l_bbox[0] for l_bbox in yolo_bbox]
        # Count the labels for each class
        bbox_labels.append(labels)
    return bbox_labels

def count_class_instances(labels_paths):
    # Count all class indexes
    bbox_labels = {}
    class_count = Counter()
    for val_label in labels_paths :
        yolo_bbox = read_annotations_yolo(val_label)
        # Select the label from annotation and flatten the lists
        labels = [bbox[0] for bbox in yolo_bbox]
        # Count the labels for each class
        class_count.update(Counter(labels))
        bbox_labels[str(val_label.name)] = labels
        #bbox_labels[str(val_label)] = labels
    return class_count, bbox_labels

def split_train_validation(root_dir, target_path,  val_ratio = 0.1, images_folder = "images",
                           labels_folder = "labels" ,  shuffle=True, split_pattern ='_x',  balance_val = False ):

    # Reference the source directories
    images_dir = Path(root_dir) / images_folder
    labels_dir = Path(root_dir) / labels_folder
    img_suffix = str(next(Path(images_dir).rglob(f"*.*")).suffix)

    # Create the new train and validation subdirectories
    Path(target_path).mkdir(parents=True, exist_ok=True)
    val_dir =   Path(target_path).resolve() / "validation"
    train_dir = Path(target_path).resolve()  / "train"
    Path(val_dir/images_folder).mkdir(parents=True, exist_ok=True)
    Path(val_dir/labels_folder).mkdir(parents=True, exist_ok=True)
    Path(train_dir/images_folder).mkdir(parents=True, exist_ok=True)
    Path(train_dir/labels_folder).mkdir(parents=True, exist_ok=True)

    basedir = Path.cwd() # save the paths relative to base directory

    # Select all the file patterns
    unique_filenames = np.unique([str(file.name).split(split_pattern)[0] for file in images_dir.rglob('*')])

    print(f"The unique non-tile images  {len(list(unique_filenames))} "
          f"and validation image number {int(len(unique_filenames)*val_ratio)}")

    # Create the unique index to do the split
    idx = np.arange(len(unique_filenames))
    if shuffle: # in case of random validation index
        np.random.shuffle(idx)

    # Select the validation filename patterns based on the validation split index and ratio
    validation_fnames = []
    validation_labels = []

    for fname in unique_filenames[ idx[:int(len(unique_filenames)*val_ratio)] ]: # Select the validation files that contain the unique filepart
        # print("VALIDATION IMAGE NAME", fname, "\n")
        # Exclude the oversample for Validation _bal_ pattern
        # validation_fnames.extend([ list(images_dir.glob(f'{fname}*(!_bal_)*')))
        validation_fnames.extend([f for f in images_dir.glob(f'{fname}*') if '_bal_' not in f.name])
        validation_labels.extend([f for f in labels_dir.glob(f'{fname}*') if '_bal_' not in f.name])
    train_fnames = []
    for fname in unique_filenames[ idx[int(len(unique_filenames)*val_ratio):]]:
        train_fnames.extend(list(images_dir.glob(f'{fname}*')))

    with open( Path(target_path).absolute()/"train_images.txt", 'a') as t1,  open(Path(target_path).absolute()/"train_labels.txt",'w') as t2:
        for label_path in train_fnames :
            t1.write(str(Path(label_path).relative_to(basedir).with_suffix(img_suffix)) +"\n")
            t2.write(str(label_path.relative_to(basedir))+"\n")

    if balance_val:
        # Count the objects for each class and delete the class
        #validation_paths = list(Path(val_dir/ labels_folder).rglob('*.txt'))

        #Get all the labels with the corresponding image files and the counted objects dict
        class_count, class_labels = count_class_instances(validation_labels)

        # Create auxiliary structure  each class instance index with the label filename
        label_idx_fnames = np.array([[l, f] for f, lbl in class_labels.items() for l in lbl]) #[[l, i, j] for i, lbl in enumerate(class_labels) for j, l in enumerate(lbl)]

        # Select randomly N objects oof each class where N is the minimum total classs  countfor the unrepresented class
        validation_fnames = np.unique([np.random.choice(label_idx_fnames[label_idx_fnames[:,0]==str(cls), 1] , min(class_count.values())) for cls in class_count.keys()])

    with open( Path(target_path).absolute()/"validation_images.txt", 'a') as f1,  open(Path(target_path).absolute()/"validation_labels.txt",'w') as f2:
        for label_path in validation_fnames:
            f1.write(str(Path(images_dir/label_path).relative_to(basedir).with_suffix(img_suffix))+"\n")
            f2.write(str((labels_dir/label_path).relative_to(basedir))+"\n")

def split_train_validation_list(images_fnames, labels_fnames=[], target_path='', split_ratio = 0.1, shuffle=True, split_pattern ='_x', balance_val = False):

    # Select all the file patterns
    unique_filenames = np.unique([str(file.name).split(split_pattern)[0] for file in images_fnames])

    # Create the unique index to do the split
    idx = np.arange(len(unique_filenames))
    if shuffle: np.random.shuffle(idx)


    split_fnames = unique_filenames[idx[:int(len(unique_filenames) * split_ratio)]]
    train_fnames = []
    train_labels = []
    for train_id in unique_filenames[idx[int(len(unique_filenames) * split_ratio):]]:
        r = re.compile(f"{train_id}*")
        train_fnames.extend( list(filter(r.match, images_fnames)) )
        train_labels.extend( list(filter(r.match, labels_fnames)) )   # Read Note below

    save_text_file(train_fnames, 'train_tile_images.txt', path= target_path)
    save_text_file(train_labels, 'train_tile_labels.txt', path= target_path)


    # Select the validation filename patterns based on the validation split index and ratio
    validation_fnames = []
    validation_labels = []

    for name_id in split_fnames: # Select the validation files that contain the unique filepart
        validation_fnames.extend([f for f in images_fnames if (name_id in f and '_bal_' not in f)])
        validation_labels.extend([f for f in labels_fnames if (name_id in f and '_bal_' not in f)])


    if balance_val: # TODO: include data balancing
        validation_labels = balance_bbox_object_labels(validation_labels)

    save_text_file(validation_fnames, 'validation_tile_images.txt', path= target_path)
    save_text_file(validation_labels, 'validation_labels.txt', path= target_path)



def balance_bbox_object_labels(validation_labels):

    # Get all the labels with the corresponding image files and the counted objects dict
    class_count, class_labels = count_class_instances(validation_labels)
    # Create auxiliary structure  each class instance index with the label filename
    label_idx_fnames = np.array([[l, f] for f, lbl in class_labels.items() for l in
                                 lbl])  # [[l, i, j] for i, lbl in enumerate(class_labels) for j, l in enumerate(lbl)]
    # Select randomly N objects oof each class where N is the minimum total classs  count for the unrepresented class
    validation_fnames = np.unique(
        [np.random.choice(label_idx_fnames[label_idx_fnames[:, 0] == str(cls), 1], min(class_count.values())) for cls in
         class_count.keys()])
    return validation_fnames


def copy_file_new_path(fpath, target_dir):
    if Path(fpath).is_file():
        Path(target_dir).mkdir(exist_ok=True, parents=True)
        shutil.copy(fpath, target_dir/fpath.name, follow_symlinks=True)

def split_train_test(root_dir, target_path, split_ratio = 0.1, images_folder ="images", split_name='train-test',
                     labels_folder = "labels", shuffle=True, split_pattern ='_x', copy_files=False):


    # Reference the source directories
    images_dir = Path(root_dir) / images_folder
    labels_dir = Path(root_dir) / labels_folder

    print(f"Number of images found in {images_dir} is {len(list(images_dir.glob('*')))}")

    # Select all the file patterns
    unique_filenames = np.unique([str(file.stem).split(split_pattern)[0] for file in images_dir.rglob('*')])

    logging.info(f"The unique images  {len(list(unique_filenames))}"
          f"and validation image number {int(len(unique_filenames) * split_ratio)}")

    # Create the unique index to do the split
    idx = np.arange(len(unique_filenames))
    if shuffle: # in case of random validation index
        np.random.shuffle(idx)

    # Select the validation filename patterns based on the validation split index and ratio
    test_fnames = []
    test_labels = []

    for f in unique_filenames[idx[:int(len(unique_filenames) * split_ratio)]]:
        test_fnames.extend(list(images_dir.glob(f'{f}*')))
        test_labels.extend(list(labels_dir.glob(f'{f}*') ))

    train_fnames = []
    train_labels = []
    for fname in unique_filenames[idx[int(len(unique_filenames) * split_ratio):]]:
        train_fnames.extend(list(images_dir.glob(f'{fname}*')))
        train_labels.extend(list(labels_dir.glob(f'{fname}*') ))

    # Create the new train and validation subdirectories
    Path(target_path).mkdir(parents=True, exist_ok=True)

    if copy_files:
        test_dir = move_split_files(target_path, test_fnames, test_labels, split_name.split('-')[-1])
        write_split_filelist(test_dir, test_fnames, test_labels, split=split_name.split('-')[-1])#, basedir= Path(root_dir))

        train_dir = move_split_files(target_path, train_fnames, train_labels, split=split_name.split('-')[0])
        write_split_filelist(train_dir, train_fnames, train_labels, split=split_name.split('-')[0])#, basedir=Path(root_dir))

    else:
        write_split_filelist(target_path, test_fnames, test_labels, split=split_name.split('-')[-1])#, basedir= Path(root_dir))
        write_split_filelist(target_path, train_fnames, train_labels, split=split_name.split('-')[0])#, basedir=Path(root_dir))

def write_split_filelist(target_path, image_fnames, labels_fnames, split='train', basedir=''):

    with open(Path(target_path)/ f"{split}_images.txt", 'a') as f1, open(Path(target_path) / f"{split}_labels.txt", 'a') as f2:
        for img, lbl in zip(image_fnames, labels_fnames):
            f1.write(str(Path(img).relative_to(basedir)) + "\n")
            f2.write(str(Path(lbl).relative_to(basedir)) + "\n")

def move_split_files(target_path, image_fnames, labels_fnames, split='train'):
    target_dir = Path(target_path).resolve() / split
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for img_fn, lbl_fn in zip(image_fnames, labels_fnames):
        copy_file_new_path(img_fn, target_dir / "images")
        copy_file_new_path(lbl_fn, target_dir / "labels")
    return target_dir


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset", type=str, help="Source Dataset directory")
    parser.add_argument("--target_path", default="dataset", help="Target data directory to save split files")
    parser.add_argument("--images_folder", default="images", help="Subdirectory name containing the images")
    parser.add_argument("--labels_folder", default="labels", help="Subdirectory name containing the labels")
    parser.add_argument("--split_percent", default=10, type=float,
                        help="Test/Validation split normalized ratio [0,100)")
    parser.add_argument("--split_name", default="train-test", type=str,
                        help="Split name to save split files separated by '-'")
    parser.add_argument("--balance_val", default=True, type=bool,
                        help="Balance object classses count and exclude augmented data")
    parser.add_argument("--shuffle_index", default=True, help="Shuffle the validation index")
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    parser.add_argument("--copy", default=False, action="store_true", help="Tag to refactor dataset into p")
    return parser.parse_args()


if __name__==  '__main__':

    args = init_args()

    # Fix Random Seed
    np.random.seed(seed=args.seed)

    # Configure data paths
    dataset_source_path = Path(args.data_path)
    target_path = Path(args.target_path) if args.target_path is not None else dataset_source_path
    if Path(args.images_folder).is_file():
        for data_path, labels_path in zip(dataset_source_path.rglob(f'{args.images_folder}**.txt'), dataset_source_path.rglob(f'{args.labels_folder}**.txt')  ):
            fnames = zip(load_text_file(data_path), load_text_file(labels_path))
            split_train_validation_list(fnames, target_path = target_path, split_ratio=args.split_percent / 100,
                                        images_folder = args.images_folder, labels_folder = args.labels_folder,
                                        shuffle=args.shuffle_index, copy_files=False)
    else:
        print(f"Separate data {dataset_source_path}")
        split_train_test(dataset_source_path, target_path = target_path, split_ratio=args.split_percent / 100,
                          images_folder = args.images_folder, labels_folder = args.labels_folder,
                          split_name=args.split_name, shuffle=args.shuffle_index, copy_files=False)