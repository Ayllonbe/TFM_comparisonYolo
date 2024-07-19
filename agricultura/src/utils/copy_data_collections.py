import argparse
from pathlib import Path
try:
    from utils.utils_io import copy_folder
except:
    import sys, os
    source_path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')
    main_path, _ = source_path.rsplit('/',1)
    main_path, _ = main_path.rsplit('/',1)
    sys.path.append(main_path)
    from src.utils.utils_io import copy_folder

def args_parser(**kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dataset",
                        help="Source folder with files to be copied or move ")
    parser.add_argument("--subfolder", default="labels",
                        help="Folder to be moved or copied ")
    parser.add_argument("--target_path", default="resized_dataset",   help="Target folder for a new sliced dataset")
    return parser.parse_args(), kwargs

def copy_files_directory(dataset_path: Path, target_path: Path, folder:str= '', **kwargs)->None:
    collections = [x for x in dataset_path.rglob(f'*{folder}') if x.is_dir()]
    for c in collections:
        print("copyin files", c)
        copy_folder(c,target_path/c.relative_to(dataset_path), **kwargs)

if __name__==  '__main__':
    args, kwargs = args_parser()
    copy_files_directory(Path(args.dataset_path), Path(args.target_path), folder=args.subfolder)