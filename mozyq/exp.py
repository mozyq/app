from pathlib import Path

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


@dataclass
class Viewport:
    width: int
    height: int

    def __iter__(self):
        yield self.width
        yield self.height


def transition(
        *,
        img: np.ndarray,
        viewport: Viewport,
        t: Transition
):
    '''
    Offsets (0,0) mean center of image is align with center of viewport
    '''

    if len(t) == 0:
        return

    x, y, scale = t
    width, height = viewport

    crop_width = even(width / scale)
    crop_height = even(height / scale)

    h, w, _ = img.shape

    i = round(((1 + 2*y) * h - crop_height) / 2)
    j = round(((1 + 2*x) * w - crop_width) / 2)

    maxi = h - crop_height
    maxj = w - crop_width
    assert 0 <= i <= maxi, f'Bad crop {i} {x} {y} {maxi}'
    assert 0 <= j <= maxj, f'Bad crop {j} {x} {y} {maxj}'

    print('Image size:', w, h)
    print('Cropping at:', x, y, 'size:', crop_width, crop_height)

    crop = img[
        i:i + crop_height,
        j:j + crop_width]

    crop = scale_down(crop, scale)
    yield crop

    if scale < .5:
        img = scale_down(img, scale)
        t.scale /= scale

    yield from transition(
        img=img,
        viewport=viewport,
        t=t.next())


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


if __name__ == '__main__':
    NUM_TILES = 15
    UNIT = 1 / NUM_TILES
    mzqs = read_mzqs(Path('output.json'))
    tiles = mzqs[0].tiles
    grid = tiles2grid([
        read_image_lab(Path(tile))
        for tile in tiles])

    frames = Path('tmp')
    frames.mkdir(parents=True, exist_ok=True)

    v = Viewport(width=600, height=750)
    t = gen_transition(
        n=60,
        sx=7 * UNIT,
        sy=7 * UNIT,
        end_scale=UNIT)

    crops = transition(
        img=grid,
        viewport=v,
        t=t)

    for i, crop in enumerate(crops):
        path = frames / f'{i:03d}.jpg'
        write_jpeg(crop.astype(np.uint8), path)
