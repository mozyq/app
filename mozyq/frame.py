import json
from math import ceil, sqrt
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from cattrs import structure
from tqdm import tqdm

from mozyq.io import read_image_lab, write_jpeg
from mozyq.mzq import Mozyq
from mozyq.util import center_crop


def crop_zoom(master: np.ndarray, zoom: float):
    if (zoom == 1):
        return master

    assert zoom > 1, 'Zoom must be greater than 1'

    h, w, _ = master.shape

    # Calculate crop dimensions (smaller than original)
    crop_h = int(h / zoom)
    crop_w = int(w / zoom)

    crop_h += crop_h % 2  # Make even
    crop_w += crop_w % 2  # Make even

    # Calculate crop offsets (center crop)
    i = (h - crop_h) // 2
    j = (w - crop_w) // 2

    # First: Crop to smaller size
    cropped = master[i:i + crop_h, j:j + crop_w]

    # Then: Zoom (resize) back to original size
    zoomed = cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4)

    zh, zw, _ = zoomed.shape
    assert (zh == h) and (zw == w), 'Error in zoomed dimensions'
    return zoomed


def smart_scale_down_crop(
        grid: np.ndarray, *,
        grid_height: int,
        grid_width: int,
        crop_height: int,
        crop_width: int):

    out_height = crop_height
    out_width = crop_width
    grid_height += grid_height % 2  # Make even
    grid_width += grid_width % 2  # Make even

    h, *_ = grid.shape
    scale = grid_height / h

    if scale < .5:
        grid = cv2.resize(
            grid,
            (grid_width, grid_height),
            interpolation=cv2.INTER_AREA)

    else:
        crop_height = ceil(crop_height / scale)
        crop_width = ceil(crop_width / scale)

    crop = center_crop(
        img=grid,
        height=crop_height,
        width=crop_width)

    if scale >= .5:
        crop = cv2.resize(
            crop,
            (out_width, out_height),
            interpolation=cv2.INTER_AREA)

    return grid, crop


def ease(s: float, e: float, n: int = 90):
    """Generate n values from s to e with ease-out effect"""
    t = np.linspace(0, 1, n)
    t = np.where(
        t < 0.5,
        4 * t**3,
        1 - (-2 * t + 2)**3 / 2)

    return s + (e - s) * t


def step(
        *,
        master: np.ndarray,
        tiles: list[np.ndarray],
        zooms: Iterable[float]):

    d = int(sqrt(len(tiles)))
    assert d ** 2 == len(tiles), 'Tiles length must be a perfect square'

    grid = [[tiles[i * d + j]
             for j in range(d)]
            for i in range(d)]

    rows = [np.hstack(row) for row in grid]
    grid = np.vstack(rows)

    h, w, _ = grid.shape
    crop_height = h // d
    crop_width = w // d

    max_zoom = max(zooms)
    for zoom in zooms:
        alpha = (1 / zoom) * 0.3
        target = crop_zoom(master, zoom)
        scale = zoom / max_zoom
        grid, crop = smart_scale_down_crop(
            grid,
            grid_height=ceil(h * scale),
            grid_width=ceil(w * scale),
            crop_height=crop_height,
            crop_width=crop_width)

        yield (1 - alpha) * crop + alpha * target

    for beta in np.linspace(alpha, 1, 30):
        yield (1 - beta) * crop + beta * target


def frames(mzq: list[Mozyq], out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)

    i = 0
    for mozyq in mzq:
        max_zoom = int(sqrt(len(mozyq.tiles)))
        zooms = ease(max_zoom, 1)

        master = read_image_lab(mozyq.master)
        tiles = [
            read_image_lab(tile_path)
            for tile_path in mozyq.tiles]

        fs = step(
            master=master,
            tiles=tiles,
            zooms=zooms)

        for frame in tqdm(fs):
            write_jpeg(
                frame.astype(np.uint8),
                out_folder / f'{i:04d}.jpg')

            i += 1


def read_all(mzq: list[Mozyq]):
    for mozyq in mzq:
        yield read_image_lab(mozyq.master)
        for tile_path in mozyq.tiles:
            yield read_image_lab(tile_path)


if __name__ == '__main__':
    with open('output.json') as f:
        mzqs = [
            structure(mzq, Mozyq)
            for mzq in json.load(f)]

    frames(mzqs, Path('./frames'))
