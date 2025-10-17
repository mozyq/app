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


def crop_zoom(master: np.ndarray, zoom: float):
    if (zoom == 1):
        return master

    assert zoom > 1, 'Zoom must be greater than 1'

    h, w, _ = master.shape

    # Calculate crop dimensions (smaller than original)
    crop_h = int(h / zoom)
    crop_w = int(w / zoom)

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


def scale_down_crop(
        grid: np.ndarray, *,
        grid_height: int,
        grid_width: int,
        crop_height: int,
        crop_width: int):

    grid = cv2.resize(
        grid,
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA)

    i = (grid_height - crop_height) // 2
    j = (grid_width - crop_width) // 2

    return grid, grid[i:i + crop_height, j:j + crop_width]


def frames(
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
        alpha = (1 / zoom) / 2
        scale = zoom / max_zoom
        target = crop_zoom(master, zoom)
        grid, crop = scale_down_crop(
            grid,
            grid_height=ceil(h * scale),
            grid_width=ceil(w * scale),
            crop_height=crop_height,
            crop_width=crop_width)

        yield (1 - alpha) * crop + alpha * target

    for beta in np.linspace(alpha, 1, 30):
        yield (1 - beta) * crop + beta * target


def ease_out(s: float, e: float, n: int = 90, p: float = .3):
    """Generate n values from s to e with ease-out effect"""
    return np.array([
        s + (e - s) * t ** p
        for t in np.linspace(0, 1, n)])


if __name__ == '__main__':
    max_zoom = 15
    zooms = ease_out(max_zoom, 1)

    with open('output.json') as f:
        mzqs = [
            structure(mzq, Mozyq)
            for mzq in json.load(f)]

    out = Path('dbg/frames')
    out.mkdir(parents=True, exist_ok=True)

    master = read_image_lab(mzqs[0].master)
    tiles = [
        read_image_lab(tile_path)
        for tile_path in mzqs[0].tiles
    ]

    # for i, frame in enumerate(frames(master, zooms=zooms)):
    #     write_jpeg(frame.astype(np.uint8), out / f'frame_{i:03d}.jpg')

    fs = frames(
        master=master,
        tiles=tiles,
        zooms=zooms)

    for i, crop in enumerate(tqdm(fs)):
        write_jpeg(crop.astype(np.uint8), out / f'{i:03d}.jpg')
