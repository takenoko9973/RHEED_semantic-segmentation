import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import transform
from skimage import measure


def mask_to_polygons_per_class(
    mask: np.ndarray,
    class_labels: dict,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    simplify_tol: float = 10.0,
):
    shapes = []
    for class_id, class_name in class_labels.items():
        binary = (mask == class_id).astype(np.uint8)
        contours = measure.find_contours(binary, 0.5)

        for contour in contours:
            if len(contour) >= 3:
                # skimage: (y, x) 順 → (x, y)に変換し、スケール補正
                polygon = [(x * scale_x, y * scale_y) for y, x in contour]

                # ShapelyでPolygon化 + 簡略化
                poly = Polygon(polygon)
                poly = poly.simplify(tolerance=simplify_tol, preserve_topology=True)

                if not poly.is_valid or poly.is_empty:
                    continue

                simplified_coords = list(poly.exterior.coords)
                if len(simplified_coords) < 3:
                    continue

                shapes.append(
                    {
                        "label": class_name,
                        "points": polygon,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                )
    return shapes


def build_labelme_json(image_path: str | Path, shapes: list, output_path: str | Path):
    image_path = Path(image_path)
    output_path = Path(output_path)

    with Image.open(image_path) as img:
        width, height = img.size

    json_dict = {
        "version": "5.7.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageHeight": height,
        "imageWidth": width,
    }

    with output_path.open("w") as f:
        json.dump(json_dict, f, indent=4)


def generate_labelme_from_mask(
    image_path: str | Path, mask_path: str | Path, class_labels: dict, output_json_path: str | Path
):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))

    width_img, height_img = image.shape
    width_mask, height_mask = mask.shape
    scale_x = width_img / width_mask
    scale_y = height_img / height_mask

    shapes = mask_to_polygons_per_class(mask, class_labels, scale_x, scale_y)
    build_labelme_json(image_path, shapes, output_json_path)


class_labels = {1: "spot", 2: "streak", 3: "kikuchi"}

generate_labelme_from_mask(
    image_path="downloads/SC-STO-250422/expo50_gain60/CLAHE/250425_900_fil-6_O2-0/0/0.0.tiff",
    mask_path="preds/SC-STO-250422/expo50_gain60/CLAHE/250425_900_fil-6_O2-0/0/0.0.png",
    class_labels=class_labels,
    output_json_path="sample.json",
)
