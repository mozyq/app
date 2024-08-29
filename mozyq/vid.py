import json
import typing
from functools import lru_cache
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
from cattrs import structure
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop
from torio.io import StreamingMediaEncoder
from tqdm import tqdm

from mozyq.io import load_grid, load_img_any_size, safe_resize
from mozyq.mzq import Mozyq, Video
from mozyq.types import Preset

FULL_GRID_MAX_ZOOM = 2


@lru_cache(maxsize=2)
def build_full_grid(mozyq: Mozyq, size: int):
    tile_size = size // mozyq.nrow
    assert tile_size * mozyq.nrow == size

    return load_grid(mozyq.tiles, tile_size)


@lru_cache(maxsize=10)
def build_patch(
        mozyq: Mozyq,
        ij: tuple[int, int],
        zoomed_tile_size: int):

    s = slice(*ij)

    tiles = mozyq.grid[(s, s)]

    return load_grid(
        tiles.ravel().tolist(),
        zoomed_tile_size)


def build_frame_from_patch(mozyq: Mozyq, master_size: int, zoom: float):
    span = mozyq.nrow / zoom
    i, j = floor((mozyq.nrow - span) / 2), ceil((mozyq.nrow + span) / 2)
    tile_size = master_size // mozyq.nrow
    assert tile_size * mozyq.nrow == master_size

    zoomed_tile_size = ceil(tile_size * zoom)
    actual_zoom = zoomed_tile_size / tile_size
    assert actual_zoom >= zoom

    patch = build_patch(
        mozyq,
        (i, j),
        zoomed_tile_size)

    if actual_zoom > tile_size:
        patch = interpolate(
            patch[None],
            scale_factor=zoom / actual_zoom).squeeze()

    return center_crop(patch, [master_size, master_size])


def build_frame(
        mozyq: Mozyq,
        master_size: int, *,
        zoom: float,
        alpha: float):

    if zoom <= FULL_GRID_MAX_ZOOM:
        grid = build_full_grid(mozyq, master_size * FULL_GRID_MAX_ZOOM)
        grid = safe_resize(grid, master_size * zoom)
        grid = center_crop(grid, [master_size, master_size])

    else:
        grid = build_frame_from_patch(mozyq, master_size, zoom)

    master = load_img_any_size(mozyq.master, master_size)
    master_patch_size = round(master_size / zoom)
    master_patch_size += master_patch_size % 2
    master = center_crop(master, [master_patch_size, master_patch_size])
    master = safe_resize(master,  master_size)

    blend = alpha * master + (1 - alpha) * grid
    return blend.to(torch.uint8)


def zooms(start: int, steps: int, b=2, end=11, eps=.01):
    def f(x):
        return 1 + (start - 1) / b ** x

    zooms = f(np.linspace(0, end, steps))
    zooms[zooms < 1 + eps] = 1

    assert min(zooms) == 1, f'{min(zooms)=}'

    return zooms


def alphas(steps: int, p=0.8):
    return np.linspace(0, 1, steps) ** p


def build_transition(
        mozyq: Mozyq,
        master_size: int, *,
        zooms: np.ndarray,
        alphas: np.ndarray):

    assert len(zooms) == len(alphas), \
        f'len(zooms) != len(alphas) {len(zooms)} != {len(alphas)}'

    tile_size = master_size // mozyq.nrow

    assert mozyq.nrow * tile_size == master_size, \
        f'{mozyq.nrow} * {tile_size} != {master_size}'

    return (
        build_frame(mozyq, master_size, zoom=zoom, alpha=alpha)
        for zoom, alpha in zip(zooms, alphas))


def save_video(
        frames: typing.Iterable[torch.Tensor],
        video_mp4: Path,
        width: int,
        height: int,
        crf: int,
        preset: Preset):

    out_stream = StreamingMediaEncoder(video_mp4)

    out_stream.add_video_stream(
        frame_rate=30,
        width=width,
        height=height,
        encoder='libx264',
        encoder_option={
            'crf': f'{crf}',
            'preset': preset})

    with out_stream.open():
        for frame in frames:
            out_stream.write_video_chunk(0, frame[None])


def build_video(
        *,
        video_json: Path,
        video_mp4: Path,
        steps_per_transition: int,
        crf: int,
        preset: Preset):

    with open(video_json) as f:
        video = json.load(f)
        video = structure(video, Video)

    def frames():
        for mozyq in video.mozyqs:
            transition = build_transition(
                mozyq,
                video.master_size,
                zooms=zooms(mozyq.nrow, steps_per_transition),
                alphas=alphas(steps_per_transition))

            for img in transition:
                yield img

    save_video(
        tqdm(
            frames(),
            desc='Generating an amazing clip',
            total=len(video.mozyqs) * steps_per_transition),

        video_mp4,
        video.master_size,
        video.master_size,
        crf=crf,
        preset=preset)
