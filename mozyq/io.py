from functools import lru_cache
from math import ceil, log, sqrt
from pathlib import Path
from typing import Iterable, Literal, get_args

import torch
from torch import Tensor
from torchvision.io import read_image, write_jpeg
from torchvision.transforms.functional import center_crop, resize
from torchvision.utils import make_grid


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


def cache_img(img: Tensor, path: Path):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg(img, str(path))


def safe_resize(img: torch.Tensor, size: int | float):
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
    img = img.expand(3, -1, -1)
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
