import  pandas as pd
from pathlib import Path
import torch
import sys
try:
    from src.bbox_annotations.annotations_io import save_labels_yolo
    from src.bbox_annotations import annotations_io
except ModuleNotFoundError:
    project_path = Path(__file__).parent.parent.parent.as_posix()
    sys.path.append(project_path)
    from src.bbox_annotations import annotations_io



def value_counts_df(df, column_str):
    """
    Returns counted values and percentage specified from column_str name param as a pandas DataFrame

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe on which to run value_counts(), must have column `col`.
    column_str : str
        Name of column in `df` for which to generate counts

    Returns
    -------
    dict
        Returned dict will have a for each class named "count" which contains the count_values()
        for each unique value of df[column_str]. Same applies for the percentage
    """
    counts =  df[column_str].value_counts().to_dict()
    percent = dict(zip(["%-"+str(k) for k in counts.keys()], df[column_str].value_counts( normalize=True).values) )
    counts.update(percent)
    return counts#pd.DataFrame(data=counts, index=[0])



def tif_tiler(multilayer_img, shape, tile=(1280, 1280)):
    num_layers, rows, columns, channels = shape
    h, w = tile
    for l in range(num_layers):
        for y in range(0, rows, h):
            for x in range(0, columns, w):
                yield multilayer_img[l][y:y + h, x:x + w, ... ]


def in_center_roi(xc: float, yc: float, tile_width: int, tile_height: int, stride_x: int, stride_y: int) -> bool:
    """
    Parameters
    ----------
    xc: float
        center x coordinate of the bounding box
    yc: float
        center y coordinate of the bounding box
    tile_width: int
    tile_height: int
    stride_x: int
    stride_y: int

    Returns
    -------
    in_roi: bool
        indicate if the coordinates are in the region of interest
    """
    center_roi = [stride_x // 2, stride_y // 2, (tile_width - stride_x // 2), (tile_height - stride_y // 2)]
    in_roi = False
    if xc >= center_roi[0] and yc >= center_roi[1] and xc < center_roi[-2] and yc < center_roi[-1]:
        in_roi = True
    return in_roi



def filter_results_yolov5(model_prediction, tile_size, stride_size=[0,0], pad=[0,0], batch = 0):

    tile_width, tile_height = tile_size
    stride_x, stride_y = stride_size
    preds = [None]*len(model_prediction)
    preds_df = []
    indx = []
    list_pandas = list()
   # results_df = model_prediction.pandas().xyxy
    for i, pred in enumerate(model_prediction):
        df = pd.DataFrame(pred.boxes.xyxy.numpy(),columns=["xmin","ymin","xmax","ymax"])
        df["confidence"] = pred.boxes.conf.numpy()
        df["class"] = pred.boxes.cls.numpy().astype(int)
        df["name"] = [pred.names[i] for i in df["class"]]
        list_pandas.append(df)         
    for i, pred in enumerate(model_prediction):

        p = 0
        preds[i] = [None] * len(pred)
        

        for det, (r,row) in zip(pred, list_pandas[i].iterrows()): #(xyxy, conf, cls)
            
            xc = ((det.boxes.data.numpy()[0][0]+det.boxes.data.numpy()[0][2])/2)
            yc = ((det.boxes.data.numpy()[0][1]+det.boxes.data.numpy()[0][3])/2)
            
            if in_center_roi(xc, yc, tile_width, tile_height, stride_x, stride_y):
                preds[i][p] = row.values  # [x1,y1,x2,y2,conf,cls,name]
                indx.append(f"{str(batch).zfill(2)}-{str(i).zfill(2)}-{str(p).zfill(2)}")

                x1,y1,x2,y2,conf,cls,name = row.values

                x1 = x1 + i*(tile_size[1]- stride_size[1]) - pad[1]//2
                x2 = x2 + i*(tile_size[1]- stride_size[1]) - pad[1]//2
                y1 = y1 + batch*(tile_size[0]- stride_size[0]) - pad[0]//2
                y2 = y2 + batch*(tile_size[0]- stride_size[0]) - pad[0]//2

                preds_df.append([x1,y1,x2,y2,conf,cls,name])


                p += 1

    return preds, pd.DataFrame(preds_df, index = indx, columns=["xmin","ymin","xmax","ymax","confidence","class","name"])


def save_preds_txt(bbox_preds, image_fname, image_size, target_path=''):
    if len(image_size)>2:
        image_size = image_size[-3:-1][::-1]
    bboxes_yolo =[]
    lbl_fname = Path(target_path)/Path(image_fname.stem).with_suffix('.txt')
    for bbox in bbox_preds:
        if bbox is not None:
            x1, y1, x2, y2, conf, lbl, name = bbox
            # Yolo Label Format: [class_id, x_center, y_center, width_normalized, height_normalized]
            bboxes_yolo.append( [lbl, x1, y1, x2, y2] )
            annotations_io.save_labels_yolo(bboxes_yolo, lbl_fname, image_size)

if __name__==  '__main__':
    pass