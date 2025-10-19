from contextlib import contextmanager
from enum import IntEnum
from time import perf_counter

import cv2
import numpy as np


class Interpolation(IntEnum):
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4


@contextmanager
def timer(label=""):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{label} took {(end - start)*1000:.2f} ms")


def _scale(
        *,
        img: np.ndarray,
        scale: float,
        inter: Interpolation):

    if scale == 1:
        return img

    h, w, _ = img.shape
    nw = even(w * scale)
    nh = even(h * scale)

    return cv2.resize(img, (nw, nh), interpolation=inter)


def scale_down(
        img: np.ndarray,
        scale: float):

    assert 0 < scale <= 1, "Scale must be < 1"

    return _scale(
        img=img,
        scale=scale,
        inter=Interpolation.AREA)


def scale_up(
        img: np.ndarray,
        scale: float):

    assert scale >= 1, "Scale must be >= 1"

    return _scale(
        img=img,
        scale=scale,
        inter=Interpolation.LANCZOS4)


def center_crop(
        *,
        img: np.ndarray,
        height: int,
        width: int):

    height -= height % 2  # Make even
    width -= width % 2  # Make even
    h, w, _ = img.shape

    assert (height <= h) and (width <= w), \
        f'Crop size must be <= image size: {height}x{width} <= {h}x{w}'

    i = (h - height) // 2
    j = (w - width) // 2
    return img[i:i + height, j:j + width]


def even(val: int | float):
    val = round(val)
    return val - (val % 2)


def tiles2grid(tiles: list[np.ndarray]):
    d = int(np.sqrt(len(tiles)))

    assert d ** 2 == len(
        tiles), f'Tiles length must be a perfect square {len(tiles)}'

    rows = [[tiles[i * d + j]
             for j in range(d)]
            for i in range(d)]

    rows = [np.hstack(row) for row in rows]
    return np.vstack(rows)
