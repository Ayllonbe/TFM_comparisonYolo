from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageColor, ImageFont
random.seed(10)
try:
    FONT = ImageFont.truetype("Arial.ttf", 20, encoding="unic")
except:
    FONT = ImageFont.load_default()

#Create the list of colors
PIL_COLORS =  random.sample(list(ImageColor.colormap.keys()), len(ImageColor.colormap.keys()))
class_colors = ["red","green","blue","cyan", "fuchsia", "limegreen", "magenta"] + PIL_COLORS


def draw_bbox(draw, bbox_preds,  colors =class_colors):
    x1, y1, x2, y2, conf, lbl, name = bbox_preds
    draw.rectangle([(x1, y1), (x2, y2)], outline=colors[lbl], width=3)
    fs = FONT.getsize(f"{name} {conf:.2f}")
    draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1 + 1], fill=colors[lbl])
    draw.text((x1, y1), f"{name} {conf:.2f}", anchor='ls', fill=(255, 255, 255), font=FONT)

def plot_full_image_bbox(image, bbox_preds_df, target_path, im_fname,**kwargs):
    colors = kwargs.get("colors", class_colors)
    img = PIL.Image.fromarray(image).convert("RGBA")
    draw = PIL.ImageDraw.Draw(img)
    for (p,bbox) in  bbox_preds_df.iterrows():
        draw_bbox(draw,bbox, colors)
    img.save(target_path/Path(im_fname).with_suffix(".png"))

def plot_tile_bbox_results(image_tiles, bbox_preds, target_path, im_fname, stride=[0, 0],batch = 0, **kwargs):
    colors = kwargs.get("colors", class_colors)
    for i, tile in enumerate(image_tiles):
        img = PIL.Image.fromarray(tile).convert("RGBA")
        draw = PIL.ImageDraw.Draw(img)
        for bbox in bbox_preds[i]:
            if bbox is not None:
                draw_bbox(draw,bbox, colors)
        # Crop the tiles to only display the non Overlapping region
        width, height = img.size
        img = img.crop((stride[0]//3,stride[1]//3, width- stride[0]//3, height - stride[1]//3 ))
        img.save(target_path/Path(im_fname + "-tile-"+ str(i*batch).zfill(3)).with_suffix(".png"))

def plot_multilayer_image_bbox(image, bbox_preds_df, target_path, im_fname,colors=class_colors, n_layers = [4,5],  tile_shape = (1280,1280)):
    frames=[]
    image_shape = image.shape
    dtype = image.dtype
    for i, page in enumerate(image[n_layers[0]:n_layers[1]]):
        page =Image.fromarray(page)
        draw = ImageDraw.Draw(page)
        for p,bbox in bbox_preds_df.iterrows():
            draw_bbox(draw,bbox, colors)
        frames.append(np.array(page.im))
    del image
    with tifffile.TiffWriter(target_path/Path(im_fname).with_suffix(".tiff"), bigtiff=True) as tif:
        tif.write( data=tif_tiler(frames,(len(frames),) + image_shape[1:],  tile_shape), shape=(len(frames),) + image_shape[1:],
                   dtype=dtype, tile=tile_shape)


def plot_tile_bbox_results(image_tiles, bbox_preds, target_path, im_fname, stride=[0, 0], colors=class_colors, batch = 0):
    for i, tile in enumerate(image_tiles):
        img = Image.fromarray(tile).convert("RGBA")
        draw = ImageDraw.Draw(img)
        for bbox in bbox_preds[i]:
            if bbox is not None:
                x1, y1, x2, y2, conf, lbl, name = bbox
                font = ImageFont.truetype("Arial.ttf", 20, encoding="unic")
                draw.rectangle([(x1, y1), (x2, y2)], outline=colors[lbl], width=3)
                fs= font.getsize(f"{name} {conf:.2f}")
                draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1  + 1], fill=colors[lbl])
                draw.text((x1, y1), f"{name} {conf:.2f}", anchor='ls', fill=(255, 255, 255), font= font )

                # Crop the tiles to only display the non Overlapping region
        width, height = img.size
        img = img.crop((stride[0]//2,stride[1]//2, width- stride[0]//2, height - stride[1]//2 ))
        img.save(target_path/Path(im_fname + "-tile-"+ str(i*batch).zfill(3)).with_suffix(".png"))


def plot_multilayer_image_bbox(image, bbox_preds_df, target_path, im_fname,colors=class_colors, n_layers = [5,6]):

    frames=[]
    font = ImageFont.truetype("Arial.ttf", 20, encoding="unic")
    for i, page in enumerate(image[n_layers[0]:n_layers[1]]):
        page =Image.fromarray(page).convert("RGBA")
        draw = ImageDraw.Draw(page)
        for (p,bbox) in  bbox_preds_df.iterrows():
            x1, y1, x2, y2, conf, lbl, name = bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=colors[lbl], width=3)
            fs= font.getsize(f"{name} {conf:.2f}")
            draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1  + 1], fill=colors[lbl])
            draw.text((x1, y1), f"{name} {conf:.2f}", anchor='ls', font= font, fill=(255, 255, 255) )
        frames.append(page)
    img =frames[0]
    img.save(target_path/Path(im_fname).with_suffix(".tiff") , save_all=True, append_images=frames[1:])

def plot_full_image_bbox(image, bbox_preds_df, target_path, im_fname,colors=class_colors):

    img = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for (p,bbox) in  bbox_preds_df.iterrows():

        x1, y1, x2, y2, conf, lbl, name = bbox
        font = ImageFont.truetype("Arial.ttf", 20, encoding="unic")

        draw.rectangle([(x1, y1), (x2, y2)], outline=colors[lbl], width=3)
        fs= font.getsize(f"{name} {conf:.2f}")
        draw.rectangle([x1, y1 - fs[1], x1 + fs[0] + 1, y1  + 1], fill=colors[lbl])
        draw.text((x1, y1), f"{name} {conf:.2f}", anchor='ls', fill=(255, 255, 255), font= font )

    img.save(target_path/Path(im_fname).with_suffix(".tiff"))
