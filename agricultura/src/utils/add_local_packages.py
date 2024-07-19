from pathlib import Path
import sys

if __name__=='__main':

    # Add the main project folder path to the sys.path list
    source_path = Path(__file__).parent.parent
    sys.path.append(source_path.name)
    print("Add local packages to Path")
    for p in source_path.rglob('**/'):
        sys.path.append(p.relative_to(source_path.parent.as_posix()))
