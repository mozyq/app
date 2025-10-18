from contextlib import contextmanager
from time import perf_counter

import cv2
import numpy as np


@contextmanager
def timer(label=""):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{label} took {(end - start)*1000:.2f} ms")


def scale_down(img: np.ndarray, scale: float):
    if scale == 1:
        return img

    assert 0 < scale < 1, "Scale must be in (0, 1) range"

    h, w, _ = img.shape
    nw = even(w * scale)
    nh = even(h * scale)

    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


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
    val = int(val)
    return val + (val % 2)


def tiles2grid(tiles: list[np.ndarray]):
    d = int(np.sqrt(len(tiles)))

    assert d ** 2 == len(tiles), 'Tiles length must be a perfect square'

    rows = [[tiles[i * d + j]
             for j in range(d)]
            for i in range(d)]

    rows = [np.hstack(row) for row in rows]
    return np.vstack(rows)
