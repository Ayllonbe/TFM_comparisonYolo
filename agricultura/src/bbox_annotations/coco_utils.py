import argparse
import shutil
import funcy
from src.utils.utils_io import extract_zip_files, load_json, save_json
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories, check_annotation=False):
    """ Function to save a COCO compatible format labels """

    if not check_annotation or annotations != []:
        # Make sure that the parent path exists or create it
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        #Save COCO formatted JSON
        with open(file, 'wt', encoding='UTF-8') as coco:
            json.dump({ 'info': info, 'licenses': licenses, 'images': images,
                        'bbox_annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)


def filter_coco_annotations(annotations, images):
    """ Function to save filter COCO images instances based in tge  """
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_coco_images(images, annotations):
    """ Function to save filter COCO labels based in bbox_annotations image_id field """
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def split_coco_labels_collection(args, f, collection):
    with open(f, 'rt', encoding='UTF-8') as annotations_file:
        coco = json.load(annotations_file)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['bbox_annotations']
        categories = coco['categories']

        for trial in collection:
            save_dir = Path('annotated_dataset') / trial / 'bbox_annotations'
            save_dir.mkdir(parents=True, exist_ok=True)
            trial_images = [img for img in images if trial in img['file_name']]
            if trial_images and args.rel_rename:
                for i, img in enumerate(trial_images):
                    p = str(img['file_name']).replace(args.rel_rename, '')
                    trial_images[i]['file_name'] = p
                annotations_trial = filter_coco_annotations(annotations, trial_images)
                save_coco(save_dir / f.name, info, licenses, trial_images, annotations_trial, categories)


def process_coco_labels(coco_fn, replace_path='', save_dir=None, **kwargs):
    """ Load a COCO JSON bbox_annotations, refactor the image path and save it in the corresponding dir

    Parameters
    ----------
    coco_fn: str, Path
        File name of COCO JSON file to load.
    replace_path: str
        Relative path or string to be removed or substitute by kwargs.new_path
    save_dir: str, Path (optional)
        Target directory Path where to save the file
    ** kwargs: keyword arguments
        - 'new_path': the new string to be substituted with
        - 'keep_structure': boolen indicating if save the annotation keeeping the image paths structures

    Returns
    ----------
    coco_annotations: Object
        COCO bbox_annotations as a JSON Object.
    """

    # Load COCO bbox_annotations
    coco_annotations = load_json(coco_fn)
    folder_structure = set([])
    if replace_path:  # Modify the image path
        for i, img in enumerate(coco_annotations['images']):
            p = Path( str(img['file_name']).replace(replace_path, kwargs.get('new_path', '') ) )
            coco_annotations['images'][i]['file_name'] =  f"{Path(p.parent).as_posix()}/images/{p.name}" if "images" not in f"{p.parent}" else p.as_posix()
            folder_structure.update([str(Path(p).parent)])

    if save_dir: # Save the COCO bbox_annotations to specified dir maintaining original name

        if kwargs.get('keep_structure', False):
            for folder in folder_structure:
                print("Saving bbox_annotations:", folder)
                save_path =  Path(save_dir) / Path( folder) / "bbox_annotations"
                coco_subset = filter_by_image_coco(coco_annotations, pattern=str(Path(folder).name))
                save_coco( save_path / Path(coco_fn).name, coco_subset['info'], coco_subset['licenses'],
                           coco_subset['images'],coco_subset['bbox_annotations'], coco_subset['categories'], check_annotation=True)
        else:
            save_path = Path(save_dir)
            save_coco(save_path / Path(coco_fn).name, info = coco_annotations['info'], licenses = coco_annotations['licenses'],
                      images= coco_annotations['images'], annotations = coco_annotations['bbox_annotations'], categories = coco_annotations['categories'])
    return coco_annotations


def filter_by_image_coco(coco_annotations, pattern=""):
    coco_filtered = coco_annotations.copy()
    coco_filtered['images'] = [img for img in coco_annotations['images'] if pattern in img['file_name']]
    coco_filtered['bbox_annotations'] = filter_coco_annotations(coco_annotations['bbox_annotations'], coco_filtered['images'] )
    return coco_filtered

def create_yolo_directories(dir):
    # Create YOLOv5 needed structure folders folders
    dir = Path(dir)
    for p in dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True) # make dir
    return dir


def create_bbox(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def cocojson2csv(json_annotation, save_fname="bbox_annotations.csv"):
    """ Auxiliar function to save a COCO json bbox_annotations into csv format """

    images = pd.DataFrame(json_annotation['images'])
    images.rename(columns={'id': 'image_id'}, inplace=True)
    annotations = pd.DataFrame(json_annotation['bbox_annotations'])
    # create bbox_annotations data frame
    annotations_dataframe = pd.merge(annotations, images, on='image_id')
    annotations_dataframe = annotations_dataframe[['file_name', 'bbox', 'category_id', 'area']].copy()
    annotations_dataframe.rename(columns={'file_name': 'image_path', 'category_id': 'label'}, inplace=True)
    annotations_dataframe['bbox'] = annotations_dataframe.apply(lambda x: create_bbox(x['bbox']), axis=1)
    # save the bbox_annotations as a csv file
    fn = Path(save_fname).with_suffix(".csv")
    annotations_dataframe.to_csv(fn , index=False)


def coco2yolo_labels(annotations_dir='annotated_dataset', target_dir='annotated_dataset', path_only_image=True, **kwargs):
    """ Auxiliar function to saves a COCO json bbox_annotations into yolov5 TXT format """

    save_dir = create_yolo_directories(target_dir)  # output directory
    # Read ALL COCO JSON labels
    for json_file in sorted(Path(annotations_dir).resolve().rglob('*instances*.json')):
        coco_json = load_json(json_file)

        # Create image dictionary by image_id
        images_dict = {'%g' % x['id']: x for x in coco_json['images']}

        # Create parents directory to store labels
        fn = Path(save_dir) / 'labels'
        fn.mkdir(exist_ok=True, parents=True)

        # Store CLASS-ID mapping
        # class_labels = coco_json["categories"]
        class_mapping = {str(cls["id"]-1): cls["name"] for cls in coco_json["categories"] }
        save_json(class_mapping, Path(save_dir) / 'labels_mapping.json' )

        if kwargs.get('keep_empty_labels',True) :
            [open(fn/Path(file['file_name']).with_suffix('.txt').name, mode='w').close() for file in coco_json['images']]

        # Write labels file
        for x in tqdm(coco_json['bbox_annotations'], desc=f'Annotations {json_file}'):
            img = images_dict['%g' % x['image_id']]

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            # The YOLOv5 format is  ([x_center, y_center, width, height] // image_size)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= img['width']     # normalize x
            box[[1, 3]] /= img['height']    # normalize y

            # Create Yolov5 bbox_annotations per image
            f = Path(img['file_name'])
            if kwargs.get('rel_rename', False): f = f.relative_to(kwargs.get('rel_rename'))
            if path_only_image: f = f.name

            if (box[2] > 0 and box[3] > 0):  # if w > 0 and h > 0
                cls = x['category_id'] - 1  # class
                line = cls, *(box)  # cls, box

                Path(fn / f).parent.mkdir(parents=True, exist_ok = True)
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')




def extract_coco_labels_zip(annotations_fname, save_dir, replace_path, keep_path_structure=True):
    """Helper function that extrats COCO bbox_annotations from CVAT dwnloaded ZIP
     and saves them order by folder structure if keep_path_structure"""
    coco_fn = extract_zip_files(annotations_fname, save_dir, regex_fnames='.json')

    for fn in coco_fn:
        process_coco_labels(fn, replace_path, save_dir=save_dir, keep_structure=keep_path_structure)
        shutil.rmtree(Path(fn).parent, ignore_errors = True)
