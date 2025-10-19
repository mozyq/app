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

    offX, offY, scale = t
    width, height = viewport

    crop_width = even(width / scale)
    crop_height = even(height / scale)

    h, w, _ = img.shape

    i = int((w * (1 - offX) - crop_width) // 2)
    j = int((h * (1 - offY) - crop_height) // 2)

    print('Image size:', w, h)
    print('Cropping at:', i, j, 'size:', crop_width, crop_height)

    crop = img[j:j + crop_height, i:i + crop_width]
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
        n: int = 90,
        sx: float = 0.0,
        sy: float = 0.0,
        end_scale: float):

    assert end_scale < 1.0, 'end_scale must be < 1.0'

    f = np.linspace(0, 1, n)
    x = sx * (1 - f)
    y = sy * (1 - f)
    scale = 1 - f * (1 - end_scale)

    return Transition(x=x, y=y, scale=scale)


if __name__ == '__main__':
    NUM_TILES = 15
    mzqs = read_mzqs(Path('output.json'))
    tiles = mzqs[0].tiles
    grid = tiles2grid([
        read_image_lab(Path(tile))
        for tile in tiles])

    frames = Path('tmp')
    frames.mkdir(parents=True, exist_ok=True)

    v = Viewport(width=600, height=750)
    t = gen_transition(
        n=90,
        sx=6 / NUM_TILES,
        sy=4 / NUM_TILES,
        end_scale=1 / NUM_TILES)

    crops = transition(
        img=grid,
        viewport=v,
        t=t)

    for i, crop in enumerate(crops):
        path = frames / f'{i:03d}.jpg'
        write_jpeg(crop.astype(np.uint8), path)
