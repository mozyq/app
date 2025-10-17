from contextlib import contextmanager
from time import perf_counter

import numpy as np


@contextmanager
def timer(label=""):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{label} took {(end - start)*1000:.2f} ms")


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
