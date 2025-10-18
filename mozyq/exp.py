from pathlib import Path

import numpy as np

from mozyq.io import read_image_lab, write_jpeg
from mozyq.util import even, scale_down


def transition(
        *,
        img: np.ndarray,
        viewport_width: int,
        viewport_height: int,
        offsetsX: np.ndarray,
        offsetsY: np.ndarray,
        scales: np.ndarray):
    '''
    Offsets (0,0) mean center of image is align with center of viewport
    '''

    assert len(offsetsX) == len(offsetsY) == len(scales), \
        'offsetX, offsetY, scales must have the same length'

    assert offsetsX.all() >= -1 and offsetsX.all() <= 1, \
        'offsetX must be in [-1, 1] range'

    assert offsetsY.all() >= -1 and offsetsY.all() <= 1, \
        'offsetY must be in [-1, 1] range'

    assert 0 < scales.all() <= 1, 'scales must be in (0, 1) range'

    if len(offsetsX) == 0:
        return

    offX, offY, scale = offsetsX[0], offsetsY[0], scales[0]

    crop_width = even(viewport_width / scale)
    crop_height = even(viewport_height / scale)

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
        scales /= scale

    yield from transition(
        img=img,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        offsetsX=offsetsX[1:],
        offsetsY=offsetsY[1:],
        scales=scales[1:])


if __name__ == '__main__':

    dog = read_image_lab(Path('dog.jpg'))

    frames = Path('tmp')
    frames.mkdir(parents=True, exist_ok=True)

    w = 300
    h = 300

    crops = transition(
        img=dog,
        viewport_width=w,
        viewport_height=h,
        offsetsX=np.array([0, .1, .2, .3]),
        offsetsY=np.array([0, .1, .2, .3]),
        scales=np.array([.8, .7, .6, .5]))

    for i, crop in enumerate(crops):
        path = frames / f'crop_{i:03d}.jpg'
        write_jpeg(crop.astype(np.uint8), path)
