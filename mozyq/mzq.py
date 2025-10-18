import json
from pathlib import Path

import numpy as np
from cattr import unstructure
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.util import view_as_blocks
from tqdm import tqdm

from mozyq.io import load_tiles, read_image_lab, write_jpeg
from mozyq.mozyq_types import Mozyq


class MozyqGenerator:
    def __init__(
            self, *,
            paths: list[Path],
            vecs: np.ndarray,
            tile_width: int,
            tile_height: int,
    ):

        assert vecs.ndim == 2, f'vectors must be 2D {vecs.shape}'

        _, s = vecs.shape

        vec_size = tile_width * tile_height * 3
        assert s == vec_size, \
            f'vectors must be of size {vec_size}'

        self.paths = np.array(paths)
        self.vecs = vecs
        self.tile_width = tile_width
        self.tile_height = tile_height

    @classmethod
    def from_folder(
            cls, folder: Path, *,
            tile_width: int,
            tile_height: int,):

        ps = sorted(list(folder.glob('*.jpg')))

        tiles = load_tiles(
            tqdm(ps, desc='reading tiles'),
            tile_width=tile_width,
            tile_height=tile_height)

        vecs = [
            tile.ravel().astype(np.float32)
            for tile in tqdm(tiles, desc='vectorizing tiles')]

        vecs = np.stack(vecs)

        return cls(
            paths=ps,
            vecs=vecs,
            tile_width=tile_width,
            tile_height=tile_height)

    def generate(self, master: np.ndarray):
        h, w, c = master.shape

        assert c == 3, 'master image must be LAB'
        assert h % 2 == 0, 'master image height must be even'
        assert w % 2 == 0, 'master image width must be even'
        assert master.size <= self.vecs.size, 'master image too large'

        master = master.astype(np.float32)

        targets = view_as_blocks(
            master,
            block_shape=(self.tile_height, self.tile_width, 3)
        ).reshape(-1, self.tile_height * self.tile_width * 3)

        def dist(tile, patch):
            return np.linalg.norm(tile - patch)

        # Compute distance matrix using scipy
        d = cdist(self.vecs, targets, metric=dist)
        rid, cid = linear_sum_assignment(d)

        # Sort indices
        return self.paths[rid][np.argsort(cid)]


def gen_mzq_json(
        *,
        master: Path,
        tile_folder: Path,
        width: int,
        height: int,
        num_tiles: int,
        output_json: Path,
        max_transitions: int = 50,):

    assert num_tiles % 2 == 1, 'num_tiles must be odd'
    assert width % 2 == 0, 'width must be even'
    assert height % 2 == 0, 'height must be even'
    assert width % num_tiles == 0, 'width must be divisible by num_tiles'
    assert height % num_tiles == 0, 'height must be divisible by num_tiles'

    tile_width = width // num_tiles
    tile_height = height // num_tiles

    gen = MozyqGenerator.from_folder(
        tile_folder,
        tile_width=tile_width,
        tile_height=tile_height)

    masters = set()
    mzqs: list[Mozyq] = []
    for _ in tqdm(range(max_transitions)):
        if master in masters:
            print(f'Master {master} already used, stopping generation')
            break

        masters.add(master)

        # GENERATE
        paths = gen.generate(read_image_lab(master))
        mzqs.append(Mozyq(master=master, tiles=paths.tolist()))

        # Use center tile as next master
        master = paths[len(paths) // 2]

    # WRITE JSON
    with output_json.open('w') as f:
        json.dump(unstructure(mzqs[:-1][::-1]), f)

    print(f'Wrote Mozyq JSON to {output_json}')


def blocks(
        img_jpg: Path, *,
        num_blocks: int = 5,
        output_folder: Path = Path('./blocks')):

    output_folder.mkdir(parents=True, exist_ok=True)

    img = read_image_lab(img_jpg)
    h, w, c = img.shape
    assert c == 3, 'image must be LAB'

    blocks = view_as_blocks(
        img, block_shape=(h // num_blocks, w // num_blocks, 3))

    for i, block in enumerate(blocks.reshape(-1, h // num_blocks, w // num_blocks, 3)):
        block_path = output_folder / f'block_{i:03d}.jpg'
        write_jpeg(block.astype(np.uint8), str(block_path))


if __name__ == '__main__':
    master = Path('./normalized/0518.jpg')
    tile_folder = Path('./blocks')

    gen_mzq_json(
        master=master,
        tile_folder=master.parent,
        width=600,
        height=750,
        num_tiles=15,
        output_json=Path('./output.json')
    )
