"""V4.1b JSON polygon annotation -> grayscale uint8 GT PNG.

Independent manual annotation must NOT be drawn by copying Shadow masks.
JSON format:
{
 "image": "v40b_00140.npz",
 "polygons": [
   {"label": "static", "points": [[x,y],...]},
   {"label": "dynamic", "points": [[x,y],...]}
 ]
}
Pixels start as IGNORE=255, polygons are drawn in the provided order,
so later polygons overwrite earlier regions. Required explicit labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

LABEL = {"static":0, "dynamic":1, "ignore":255}


def rasterize(image_shape, polygons):
    h,w=image_shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Invalid canvas shape")
    canvas=Image.new("L",(w,h),255)
    draw=ImageDraw.Draw(canvas)
    for poly in polygons:
        label=poly.get("label","").strip().lower()
        if label not in LABEL:
            raise ValueError("Label must be static/dynamic/ignore")
        pts=poly.get("points")
        if not isinstance(pts,list) or len(pts)<3:
            raise ValueError("Polygon must have >=3 points")
        points=[]
        for pt in pts:
            if not isinstance(pt,(list,tuple)) or len(pt)!=2:
                raise ValueError("Each vertex is [x,y]")
            x,y=float(pt[0]),float(pt[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("Nonfinite vertex")
            if not (0<=x<w and 0<=y<h):
                raise ValueError("Vertex out of image bounds")
            points.append((round(x),round(y)))
        draw.polygon(points,fill=LABEL[label])
    return np.asarray(canvas,dtype=np.uint8)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--npz",required=True,help="Original v40b_XXXXX.npz")
    p.add_argument("--json",required=True,help="Manual polygon annotation")
    p.add_argument("--output",required=True,help="Output GT 8bit PNG")
    args=p.parse_args()
    with np.load(args.npz,allow_pickle=False) as a:
        rgb=a["rgb_observed"]
    annotations=json.loads(Path(args.json).read_text(encoding="utf-8"))
    if not isinstance(annotations.get("polygons"),list):
        p.error('JSON must contain "polygons" list')
    gt=rasterize(rgb.shape,annotations["polygons"])
    if not np.any(gt!=255):
        p.error("Annotation has no labeled pixels; add static/dynamic polygons")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(gt,mode="L").save(output)
    print("Saved",output, "label histogram",
          {int(x):int((gt==x).sum()) for x in np.unique(gt)})


if __name__=="__main__":
    main()
