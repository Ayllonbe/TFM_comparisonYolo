import cv2
import json
import yaml
from ruamel.yaml import round_trip_dump
import errno
import shutil
import os
from pathlib import Path
import gzip
import json
from zipfile import ZipFile


def load_yaml(filename):
    """ Load a Python Object with from  a YAML file .yaml

    Parameters
    ----------
    filename: str, Path
        File name of the file we want to save.
    path: str, Path
        Target directory Path where to save the file
    Returns
    ----------
    data: Object
        Python object loaded be serialized into a YAML stream.
    """
    with open(filename, 'r') as fd:
        return yaml.safe_load(fd)


def save_yaml(data, filename, path=''):
    """ Save a Python Object with to disk in .yaml

    Parameters
    ----------
    data: Object
          Python objects to be serialized into a YAML stream.
    filename: str, Path
        File name of the file we want to save.
    path: str, Path
        Target directory Path where to save the file
    """
    fp = Path(path) / Path(filename).with_suffix('.yaml')
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(fp, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_yaml_indented(data, filename, path=''):
    """ Save a Python Object with to disk in .yaml

    Parameters
    ----------
    data: Object
        Python objects to be serialized into a YAML stream.
    filename: str, Path
    File name of the file we want to save.
    path: str, Path
    Target directory Path where to save the file
    """
    from ruamel.yaml import round_trip_dump

    fp = Path(path) / Path(filename).with_suffix('.yaml')
    fp.parent.mkdir(exist_ok=True, parents=True)

    with open(fp, 'w') as stream:
        round_trip_dump(data, stream, indent=4, block_seq_indent=2)


def ordered_yaml(data_dict,  order = ['']):
    """ Convert a dictionary to a YAML string with preferential ordering

    for some keys. Converted string is meant to be fairly human readable.

    Parameters
    ----------
    data_dict : dict
        Dictionary to convert to a YAML string.
    order: list of str
        List containing the desired order of the final YAML string
    Returns
    -------
    str
        Nicely formatted YAML string.

    """

    s = []
    yml = [yaml.dump({key: data_dict[key]}, default_flow_style=False, indent=4) for key in order if key in data_dict]
    yml.append([yaml.dump({key: data_dict[key]}, default_flow_style=False, indent=4) for key in data_dict if key not in order])
    return str(yml[0])


def load_params(filename ="params.yaml"):
    """ Load parameters YAML file  """
    if not os.path.exists(filename):
        raise Exception("Parameters file not found")
    return load_yaml(filename)


def load_json(filename):
    """ Load a JSON Object compressed as a binary file or path

    Parameters
    ----------
    filename: str, Path
        File name of JSON file we want to load.

    Returns
    ----------
    data: Object
        Python object loaded be serialized into a JSON Object.
    """

    with open(filename, 'rt', encoding='UTF-8') as f:
        return json.load(f)

def save_json(data, filename, path=''):
    """ Save a Python Object as JSON text file

    Parameters
    ----------
    data: Object
        Python object loaded be serialized into a JSON Object.
    filename: str, Path
        File name of JSON file we want to load.
    """
    fp = Path(path) / Path(filename).with_suffix('.json')
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(fp , 'w', encoding='UTF-8') as f:
        json.dump(data, f, sort_keys = True, indent = 4, ensure_ascii = False)


def extract_zip_folder(folder_path, extract_path='bbox_annotations' ):
    """ Load a compressed Directory and extract all teh content """

    with ZipFile(folder_path, 'r') as zip:
       zip.extractall(extract_path)


def extract_zip_files(folder_path, extract_path, regex_fnames=''):
    """ Load a compressed Directory and extract all teh content

    Parameters
    ----------
    folder_path: str, Path
        Source directory path of the ZIP folder to load.
    extract_path: str, Path
        Target Folder  path to save the extracted files
    Returns
    ----------
    extrated_fn: list
        Python list containing the extracted file's paths.
    """

    extrated_fn = []
    with ZipFile(folder_path, 'r') as zip:
        for f in  zip.namelist():
           if regex_fnames in f:
               extrated_fn.append(zip.extract(f,extract_path))
    return extrated_fn



def load_json_zip(filename):
    """ Load a JSON Object compressed as a gzip-compressed file

    Parameters
    ----------
    filename: str, Path
        File name of zipped JSON file we want to load.

    Returns
    ----------
    data: Object
        Python object loaded be serialized into a JSON Object.
    """
    with gzip.open(filename, 'rt', encoding='UTF-8') as zipfile:
        return json.load(zipfile)

def save_text_file(data, filename, path=''):
    """ Save a list of strings to disk

    Parameters
    ----------
    data: Object
          Python objects to be serialized into a text file.
    filename: str, Path
        File name of the file we want to save.
    path: str, Path
        Target directory Path where to save the file
    """
    fp = Path(path) / Path(filename).with_suffix('.txt')
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(fp, 'w') as f:
        f.writelines("%s\n" % l for l in list(data))

def load_text_file(filename):
    """ Load a text binary file as a list of strings line by line

    Parameters
    ----------
    filename: str, Path
        File name of JSON file we want to load.

    Returns
    ----------
    data: Object
        Python object loaded be serialized into a JSON Object.
    """

    lines = []
    with open(filename, 'r') as f:
        # testList = f.readlines()
        for line in f:
            # remove linebreak
            lines.append( line[:-1])
    return lines


def copy_file_new_path(fpath, target_path):
    """ Copy a single file to a new target fdirectory
    Parameters
    ----------
    fpath: str, Path
        File name of file we want to copy.
    path: str, Path
        Target directory Path where to save the file
    """
    if Path(fpath).is_file():
        Path(target_path).mkdir(exist_ok=True, parents=True)
        shutil.copy(fpath, target_path / fpath.name, follow_symlinks=True)

def copy_folder(src_path,  dst_path, **kwargs ):
    """ Recursively copy a directory tree and return the destination directory
    Parameters
    ----------
    src_path: str, Path
        File name of directory we want to copy.
    dst_path
        src_path: str, Path
        File name of directory we want to copy.
    kwargs: dict
        configurable parameter for the copy of the directory. Options:
        kwargs={symlinks:False, ignore:None, copy_function:copy2,
             ignore_dangling_symlinks:False, dirs_exist_ok:False}

    Returns
    ----------
    data: str
        destination directory if correct copied. Default ''.
    """
    target_dir = ''
    try:
       target_dir =shutil.copytree(src_path, dst_path,  dirs_exist_ok=True, **kwargs) # symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False,
    except OSError as exc:
       if exc.errno in (errno.ENOTDIR, errno.EINVAL):
           target_dir =shutil.copy2(src_path, dst_path,follow_symlinks=True)
       else: raise
    return target_dir