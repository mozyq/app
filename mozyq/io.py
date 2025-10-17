from functools import lru_cache
from math import ceil, log, sqrt
from pathlib import Path
from typing import Iterable, Literal, get_args

import cv2
import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    """Read image and convert to numpy array in CHW format with values 0-255"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image from {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert from HWC to CHW format
    img_array = np.transpose(img, (2, 0, 1)).astype(np.uint8)
    return img_array


def write_jpeg(img: np.ndarray, path: str, quality: int = 90):
    """Write numpy array image to JPEG file"""
    # Convert from CHW to HWC
    if img.ndim == 3:
        img_np = np.transpose(img, (1, 2, 0))
    else:
        img_np = img

    # Convert to PIL and save as JPEG
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    pil_img = Image.fromarray(img_np)
    pil_img.save(path, 'JPEG', quality=quality)


def center_crop(img: np.ndarray, size: list[int]) -> np.ndarray:
    """Center crop numpy array image to specified size [height, width]"""
    _, h, w = img.shape
    th, tw = size

    i = (h - th) // 2
    j = (w - tw) // 2

    return img[:, i:i+th, j:j+tw]


def resize(img: np.ndarray, size: list[int]) -> np.ndarray:
    """Resize numpy array image to specified size [height, width]"""
    # Convert from CHW to HWC for OpenCV processing
    if img.ndim == 3:
        img_np = np.transpose(img, (1, 2, 0))
    else:
        img_np = img

    # Resize using OpenCV
    resized = cv2.resize(
        img_np, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    # Convert back to CHW format
    if len(resized.shape) == 2:
        resized = resized[:, :, None]

    return np.transpose(resized, (2, 0, 1)).astype(img.dtype)


def make_grid(arrays: list[np.ndarray], nrow: int, padding: int = 0) -> np.ndarray:
    """Create a grid of images from a list of numpy arrays"""
    if not arrays:
        raise ValueError("Cannot create grid from empty array list")

    # Get dimensions
    _, h, w = arrays[0].shape
    ncol = len(arrays) // nrow

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


def safe_resize(img: np.ndarray, size: int | float):
    size = ceil(size)
    size += size % 2
    return resize(img, [size, size])


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
def load_img_any_size(path: Path, size: int):
    return safe_resize(
        load_img(path, cacheable_size(size)),
        size)


def load_tiles(paths: Iterable[Path], tile_size: int):
    tile_cs = cacheable_size(tile_size)
    tiles = [load_img(p, tile_cs) for p in paths]
    tiles = [safe_resize(tile, tile_size) for tile in tiles]

    return tiles


def load_grid(paths: list[Path], tile_size: int):
    nrow = int(sqrt(len(paths)))
    assert nrow * nrow == len(paths), \
        f'bad number of tiles {len(paths)}'

    return make_grid(
        load_tiles(paths, tile_size),
        nrow=nrow,
        padding=0)
