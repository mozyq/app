import json
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from cattrs import structure
from PIL import Image
from tqdm import tqdm

from mozyq.mozyq_types import Mozyq


def read_mzqs(mzq_json: Path):
    with open(mzq_json) as f:
        mzqs = json.load(f)
        return [structure(mzq, Mozyq) for mzq in mzqs]


@cache
def read_image_lab(path: Path) -> np.ndarray:
    """Read image and convert to LAB color space in CHW format with values 0-255"""
    img = cv2.imread(str(path))

    assert img is not None, f"Could not read image from {path}"

    # Convert BGR (OpenCV default) → LAB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img


def load_tiles(paths: list[Path], *, tw, th):
    for path in tqdm(paths, desc='Reading tiles'):
        tile = read_image_lab(path)
        yield cv2.resize(
            tile,
            (tw, th),
            interpolation=cv2.INTER_AREA)


def write_jpeg(img: np.ndarray, path: str | Path, quality: int = 90):
    """Write numpy array image to JPEG file"""
    assert img.shape[-1] == 3, 'Expecting LAB image'

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(path, 'JPEG', quality=quality)
