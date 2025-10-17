import json

import cv2
import numpy as np
from cattrs import structure

from mozyq.io import read_image_lab, write_jpeg
from mozyq.mzq import Mozyq


def zoom_crop(master: np.ndarray, zoom: float):
    if (zoom == 1):
        return master

    assert zoom > 1, 'Zoom must be greater than 1'

    h, w, _ = master.shape
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    assert new_h > h
    assert new_w > w

    i = (new_h - h) // 2
    j = (new_w - w) // 2

    master = cv2.resize(
        master,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4)

    return master[i:i + h, j:j + w]


def frame(mzq: Mozyq, zoom: float):
    return zoom_crop(
        read_image_lab(mzq.master),
        zoom=zoom)


if __name__ == '__main__':
    with open('output.json') as f:
        mzqs = [
            structure(mzq, Mozyq)
            for mzq in json.load(f)]

    f = frame(mzqs[0], zoom=1)
    write_jpeg(f.astype(np.uint8), 'frame.jpg')
