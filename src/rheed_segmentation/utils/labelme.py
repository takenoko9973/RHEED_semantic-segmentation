import math
import uuid
from typing import Any

import imgviz
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw


def shape_to_mask(
    img_shape: tuple[int, ...],
    points: list[list[float]],
    shape_type: str | None = None,
    line_width: int = 10,
    point_size: int = 5,
) -> npt.NDArray[np.bool_]:
    mask = Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"  # noqa: PLR2004, S101
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"  # noqa: PLR2004, S101
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"  # noqa: PLR2004, S101
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"  # noqa: S101
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    elif shape_type in [None, "polygon"]:
        assert len(xy) > 2, "Polygon must have points more than 2"  # noqa: PLR2004, S101
        draw.polygon(xy=xy, outline=1, fill=1)
    else:
        msg = f"shape_type={shape_type!r} is not supported."
        raise ValueError(msg)
    return np.array(mask, dtype=bool)


def shapes_to_label(
    img_shape: tuple[int, ...],
    shapes: list[dict[str, Any]],
    label_name_to_value: dict[str, int],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]

        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask: npt.NDArray[np.bool_]
        if shape_type == "mask":
            if not isinstance(shape["mask"], np.ndarray):
                msg = "shape['mask'] must be numpy.ndarray"
                raise ValueError(msg)

            mask = np.zeros(img_shape[:2], dtype=bool)
            (x1, y1), (x2, y2) = np.asarray(points).astype(int)
            mask[y1 : y2 + 1, x1 : x2 + 1] = shape["mask"]
        else:
            mask = shape_to_mask(img_shape[:2], points, shape_type)

        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def create_mask(json_data: dict, labels: dict[str, int]) -> npt.NDArray[np.int32]:
    img_shape = json_data["imageHeight"], json_data["imageWidth"]

    full_mask, _ = shapes_to_label(
        img_shape,
        sorted(json_data["shapes"], key=lambda shape: labels[shape["label"]], reverse=True),
        labels,
    )

    return full_mask


def create_masks_per_labels(
    json_data: dict, labels: dict[str, int]
) -> dict[str, npt.NDArray[np.int32]]:
    img_shape = json_data["imageHeight"], json_data["imageWidth"]

    masks = {}
    for label, label_id in labels.items():
        if label_id <= 0:
            continue

        shapes = filter(lambda shape: shape["label"] == label, json_data["shapes"])
        mask, _ = shapes_to_label(
            img_shape,
            shapes,
            {"_background_": 0, label: 1},
        )
        masks[label] = mask

    return masks


def label_colormap(n_label: int = 256, value: float | None = None) -> np.ndarray:
    """Label colormap.

    Parameters
    ----------
    n_label: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval: np.ndarray, idx: int) -> npt.NDArray[np.uint8]:
        shape = (*byteval.shape, 8)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)

    if value is not None:
        hsv = rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        elif isinstance(value, int):
            hsv[:, 1:, 2] = value
        else:
            raise ValueError
        cmap = hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert rgb to hsv.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Output hsv image.

    """
    hsv = Image.fromarray(rgb, mode="RGB")
    hsv = hsv.convert("HSV")
    return np.asarray(hsv)


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert hsv to rgb.

    Parameters
    ----------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Input hsv image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.

    """
    rgb = Image.fromarray(hsv, mode="HSV")
    rgb = rgb.convert("RGB")
    return np.asarray(rgb)


def convert_color_lbl(lbl: np.ndarray) -> Image.Image:
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 0xFF:  # noqa: PLR2004
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        return lbl_pil

    msg = "[%s] Cannot save the pixel-wise class label as PNG. "
    raise ValueError(msg)
