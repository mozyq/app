from pathlib import Path

import cv2
import numpy as np
from attr import dataclass

from mozyq.io import read_image_lab
from mozyq.mozyq_types import Mozyq
from mozyq.util import even, scale_down, tiles2grid


@dataclass
class Transition:
    x: np.ndarray
    y: np.ndarray
    scale: np.ndarray

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        yield self.x[0]
        yield self.y[0]
        yield self.scale[0]

    def next(self):
        self.x = self.x[1:]
        self.y = self.y[1:]
        self.scale = self.scale[1:]

        return self

    def clone(self):
        return Transition(
            x=self.x.copy(),
            y=self.y.copy(),
            scale=self.scale.copy())


@dataclass
class Viewport:
    width: int
    height: int

    def __iter__(self):
        yield self.width
        yield self.height


def _master_transition(
        *,
        master: np.ndarray,
        max_zoom: float,
        t: Transition):

    if len(t) == 0:
        return

    x, y, scale = t

    h, w, _ = master.shape

    zoom = max(1, max_zoom * scale)
    crop_width = even(w / zoom)
    crop_height = even(h / zoom)

    i = round(((1 + 2*y) * h - crop_height) / 2)
    j = round(((1 + 2*x) * w - crop_width) / 2)

    maxi = h - crop_height
    maxj = w - crop_width
    assert 0 <= i <= maxi, f'Bad crop {i} {x} {y} {maxi}'
    assert 0 <= j <= maxj, f'Bad crop {j} {x} {y} {maxj}'

    crop = master[
        i:i + crop_height,
        j:j + crop_width]

    crop = cv2.resize(
        crop,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4)

    yield crop
    yield from _master_transition(
        master=master,
        max_zoom=max_zoom,
        t=t.next())


def _grid_transition(
        *,
        grid: np.ndarray,
        viewport: Viewport,
        t: Transition
):
    if len(t) == 0:
        return

    x, y, scale = t
    width, height = viewport

    crop_width = even(width / scale)
    crop_height = even(height / scale)

    h, w, _ = grid.shape

    i = round(((1 + 2*y) * h - crop_height) / 2)
    j = round(((1 + 2*x) * w - crop_width) / 2)

    maxi = h - crop_height
    maxj = w - crop_width
    assert 0 <= i <= maxi, f'Bad crop {i} {x} {y} {maxi}'
    assert 0 <= j <= maxj, f'Bad crop {j} {x} {y} {maxj}'

    crop = grid[
        i:i + crop_height,
        j:j + crop_width]

    crop = cv2.resize(
        crop,
        (width, height),
        interpolation=cv2.INTER_AREA)

    yield crop

    if scale < .5:
        grid = scale_down(grid, scale)
        t.scale /= scale

    yield from _grid_transition(
        grid=grid,
        viewport=viewport,
        t=t.next())


def _gen_transition(
        *,
        fpt: int,
        sx: float = 0.0,
        sy: float = 0.0,
        end_scale: float):

    assert end_scale < 1.0, 'end_scale must be < 1.0'

    f = np.linspace(0, 1, fpt)
    f = 0.5 * (1 - np.cos(np.pi * f))

    x = sx * (1 - f)
    y = sy * (1 - f)
    scale = 1 - f * (1 - end_scale)

    eps = 1e-6
    assert np.all((-.5 <= x) & (x <= .5)), 'x out of bounds'
    assert np.all((-.5 <= y) & (y <= .5)), 'y out of bounds'
    assert np.all(
        (end_scale - eps <= scale)
        & (scale <= 1)), 'scale out of bounds'

    return Transition(x=x, y=y, scale=scale)


def _transition(
        master: np.ndarray,
        grid: np.ndarray,
        t: Transition):

    h, w, _ = master.shape
    gh, gw, _ = grid.shape
    assert gh / h == gw / w, 'Aspect ratio of master and grid must match'
    max_zoom = gw / w
    vp = Viewport(
        width=w,
        height=h)

    b = 0.3
    blend = np.linspace(0, b, len(t))
    for a, crop_grid, crop_master in zip(
            blend,
            _grid_transition(
                grid=grid,
                viewport=vp,
                t=t),

            _master_transition(
                master=master,
                max_zoom=max_zoom,
                t=t.clone())):

        yield a * crop_master + (1 - a) * crop_grid

    fade = b + (1 - b) * np.linspace(0, 1, 30) ** 2
    for a in fade:
        yield a * master + (1 - a) * crop_grid


def mzq_transition(mzq: Mozyq, fpt: int):
    master = read_image_lab(Path(mzq.master))
    grid = tiles2grid([
        read_image_lab(Path(tile))
        for tile in mzq.tiles])

    dim = int(np.sqrt(len(mzq.tiles)))
    s = mzq.start
    row, col = divmod(s, dim)
    t = _gen_transition(
        fpt=fpt,
        sx=(col - (dim - 1) / 2) / dim,
        sy=(row - (dim - 1) / 2) / dim,
        end_scale=1 / dim)

    yield from _transition(
        master=master,
        grid=grid,
        t=t)
