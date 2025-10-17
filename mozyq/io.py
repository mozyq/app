from functools import lru_cache
from math import ceil, log
from pathlib import Path
from typing import Iterable, Literal, get_args

import cv2
import numpy as np
from PIL import Image


def read_image_lab(path: Path) -> np.ndarray:
    """Read image and convert to LAB color space in CHW format with values 0-255"""
    img = cv2.imread(str(path))

    assert img is not None, f"Could not read image from {path}"

    # Convert BGR (OpenCV default) → LAB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img


def resize(
        img: np.ndarray, *,
        width: int,
        height: int) -> np.ndarray:
    """Resize numpy array image to specified size [height, width]"""
    assert img.ndim == 3 and img.shape[-1] == 3, 'Expecting HWC image'

    h, w, _ = img.shape
    inter = cv2.INTER_LANCZOS4 if (width < w or height < h) else cv2.INTER_AREA

    # Resize using OpenCV
    return cv2.resize(
        img,
        (width, height),
        interpolation=inter)


def center_crop(img: np.ndarray, size: list[int]) -> np.ndarray:
    """Center crop numpy array image to specified size [height, width]"""
    _, h, w = img.shape
    th, tw = size

    i = (h - th) // 2
    j = (w - tw) // 2

    return img[:, i:i+th, j:j+tw]


def load_tiles(
        paths: Iterable[Path], *,
        tile_width: int,
        tile_height: int):

    dbg = Path('dbg')
    dbg.mkdir(parents=True, exist_ok=True)

    for path in paths:
        tile = read_image_lab(path)
        yield resize(tile, width=tile_width, height=tile_height)


# TODO
def write_jpeg(img: np.ndarray, path: str | Path, quality: int = 90):
    """Write numpy array image to JPEG file"""
    assert img.shape[-1] == 3, 'Expecting LAB image'

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(path, 'JPEG', quality=quality)


def make_grid(arrays: list[np.ndarray], nrow: int, ncol: int | None = None, padding: int = 0) -> np.ndarray:
    """Create a grid of images from a list of numpy arrays"""
    if not arrays:
        raise ValueError("Cannot create grid from empty array list")

    if ncol is None:
        ncol = len(arrays) // nrow

    assert len(arrays) == nrow * ncol, \
        f"Number of arrays {len(arrays)} doesn't match grid size {nrow}x{ncol}"

    # Get dimensions
    _, h, w = arrays[0].shape

    # Create output array
    grid_h = nrow * h + (nrow - 1) * padding
    grid_w = ncol * w + (ncol - 1) * padding
    grid = np.zeros((3, grid_h, grid_w), dtype=arrays[0].dtype)

    # Place images in grid
    for i, array in enumerate(arrays):
        row = i // ncol
        col = i % ncol

        y_start = row * (h + padding)
        x_start = col * (w + padding)

        grid[:, y_start:y_start+h, x_start:x_start+w] = array

    return grid


class fs:
    photos = Path('photos')
    cache = Path('.cache')


CacheableSize = Literal[2048, 1024, 512, 256, 128, 64, 32]


def cacheable_size(size: int | float) -> CacheableSize:
    size = ceil(size)
    return 2 ** ceil(log(size, 2))


def cache_path(size: CacheableSize, name: str):
    assert size in get_args(CacheableSize), \
        f'bad cache size {size}'

    return fs.cache / str(size) / name


def cache_img(img: np.ndarray, path: Path):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg(img, str(path))


def safe_resize(
        img: np.ndarray, *,
        width: int,
        height: int):
    '''Resize to even dimensions'''
    width += width % 2
    height += height % 2
    return resize(img, [height, width])


@lru_cache(maxsize=500)
def load_img(path: Path, size: CacheableSize):
    assert size in get_args(CacheableSize), \
        f'bad cache size {size}'

    cache = cache_path(size, path.name)
    if cache.exists():
        return read_image(str(cache))

    img = read_image(str(path))
    _, h, w = img.shape
    d = min(h, w)

    img = center_crop(img, [d, d])
    img = safe_resize(img, size)
    # Ensure 3 channels (RGB)
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    cache_img(img, cache)

    return img


@lru_cache(maxsize=500)
def load_img_any_size(path: Path, width: int, height: int | None = None):
    """Load image with arbitrary dimensions"""
    if height is None:
        # Square image
        return safe_resize(
            load_img(path, cacheable_size(width)),
            width)
    else:
        # Rectangular image
        base_size = max(width, height)
        img = load_img(path, cacheable_size(base_size))
        return safe_resize(img, width, height)


# def load_tiles(
#         paths: Iterable[Path], *,
#         tile_width: int,
#         tile_height: int):

#     tile_cs = cacheable_size(tile_width)
#     tiles = [load_img(p, tile_cs) for p in paths]
#     tiles = [
#         safe_resize(
#             tile,
#             width=tile_width,
#             height=tile_height)
#         for tile in tiles]

#     return tiles


def load_grid(paths: list[Path], tile_size: int, grid_shape: tuple[int, int] | None = None):
    if grid_shape is None:
        # Default to square grid
        nrow = int(len(paths) ** 0.5)
        assert nrow * nrow == len(paths), \
            f'bad number of tiles {len(paths)} for square grid'
        ncol = nrow
    else:
        nrow, ncol = grid_shape
        assert nrow * ncol == len(paths), \
            f'bad number of tiles {len(paths)} for grid shape {nrow}x{ncol}'

    return make_grid(
        load_tiles(paths, tile_size),
        nrow=nrow,
        ncol=ncol,
        padding=0)


if __name__ == '__main__':
    tiles = Path('blocks').glob('*.jpg')
    load_tiles(
        tiles,
        tile_width=150,
        tile_height=120
    )
