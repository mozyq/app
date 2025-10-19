from pathlib import Path

import cv2
import numpy as np
from attr import dataclass

from mozyq.io import read_image_lab, read_mzqs, write_jpeg
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


def gen_transition(
        *,
        n: int = 30,
        sx: float = 0.0,
        sy: float = 0.0,
        end_scale: float):

    assert end_scale < 1.0, 'end_scale must be < 1.0'

    f = np.linspace(0, 1, n)
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


def master_transition(
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

    print('Zooming:', zoom, 'Scale:', scale)
    print('Cropping at:', i, j, 'size:', crop_width, crop_height)

    crop = master[
        i:i + crop_height,
        j:j + crop_width]

    crop = cv2.resize(
        crop,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4)

    yield crop
    yield from master_transition(
        master=master,
        max_zoom=max_zoom,
        t=t.next())


def grid_transition(
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

    print('Grid size:', w, h)
    print('Cropping at:', i, j, 'size:', crop_width, crop_height)

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

    yield from grid_transition(
        grid=grid,
        viewport=viewport,
        t=t.next())


def transition(
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

    blend = np.linspace(0, .4, len(t))
    for a, crop_grid, crop_master in zip(
            blend,
            grid_transition(
                grid=grid,
                viewport=vp,
                t=t),

            master_transition(
                master=master,
                max_zoom=max_zoom,
                t=t.clone())):
        yield a * crop_master + (1 - a) * crop_grid


if __name__ == '__main__':
    NUM_TILES = 15
    UNIT = 1 / NUM_TILES
    mzqs = read_mzqs(Path('output.json'))
    tiles = mzqs[0].tiles

    master = read_image_lab(Path(mzqs[0].master))
    grid = tiles2grid([
        read_image_lab(Path(tile))
        for tile in tiles])

    frames = Path('tmp')
    frames.mkdir(parents=True, exist_ok=True)

    # v = Viewport(width=600, height=750)
    t = gen_transition(
        n=60,
        sx=-7 * UNIT,
        sy=-7 * UNIT,
        end_scale=UNIT)

    # crops = grid_transition(
    #     grid=grid,
    #     viewport=v,
    #     t=t)

    # crops = master_transition(
    #     master=master,
    #     max_zoom=15,
    #     t=t)

    crops = transition(
        master=master,
        grid=grid,
        t=t)

    for i, crop in enumerate(crops):
        path = frames / f'{i:03d}.jpg'
        write_jpeg(crop.astype(np.uint8), path)
