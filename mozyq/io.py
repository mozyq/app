from functools import cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


@cache
def read_image_lab(path: Path) -> np.ndarray:
    """Read image and convert to LAB color space in CHW format with values 0-255"""
    img = cv2.imread(str(path))

    assert img is not None, f"Could not read image from {path}"

    # Convert BGR (OpenCV default) → LAB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img


def scale_down(
        img: np.ndarray, *,
        width: int,
        height: int) -> np.ndarray:
    """Resize numpy array image to specified size [height, width]"""
    assert img.ndim == 3 and img.shape[-1] == 3, 'Expecting HWC image'

    # Resize using OpenCV
    return cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_AREA)


def load_tiles(
        paths: Iterable[Path], *,
        tile_width: int,
        tile_height: int):

    dbg = Path('dbg')
    dbg.mkdir(parents=True, exist_ok=True)

    for path in paths:
        tile = read_image_lab(path)
        yield scale_down(tile, width=tile_width, height=tile_height)


def write_jpeg(img: np.ndarray, path: str | Path, quality: int = 90):
    """Write numpy array image to JPEG file"""
    assert img.shape[-1] == 3, 'Expecting LAB image'

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(path, 'JPEG', quality=quality)
