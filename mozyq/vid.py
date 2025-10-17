import json
import os
import tempfile
import typing
from functools import lru_cache
from math import ceil, floor
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from cattrs import structure
from tqdm import tqdm

from mozyq.io import center_crop, load_grid, load_img_any_size, safe_resize
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
        # Implement interpolation using cv2.resize
        scale_factor = zoom / actual_zoom
        new_size = (int(patch.shape[2] * scale_factor),
                    int(patch.shape[1] * scale_factor))
        # Convert CHW to HWC for cv2
        patch_hwc = np.transpose(patch, (1, 2, 0))
        resized_hwc = cv2.resize(
            patch_hwc, new_size, interpolation=cv2.INTER_LINEAR)
        # Convert back to CHW
        patch = np.transpose(resized_hwc, (2, 0, 1))

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
    return blend.astype(np.uint8)


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
        frames: typing.Iterable[np.ndarray],
        video_mp4: Path,
        width: int,
        height: int,
        crf: int,
        preset: Preset):

    # Create temporary directory for frame images
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = []

        # Save each frame as an image
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            frame_paths.append(frame_path)

            # Convert numpy array from CHW to HWC
            frame_np = np.transpose(frame, (1, 2, 0))

            # Save using OpenCV
            import cv2
            cv2.imwrite(frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

        # Create video from frames using ffmpeg
        if frame_paths:
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            (
                ffmpeg
                .input(input_pattern, framerate=30)
                .output(str(video_mp4), vcodec='libx264', crf=crf, preset=preset)
                .overwrite_output()
                .run(quiet=True)
            )


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
