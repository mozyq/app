import json
import random
from pathlib import Path

import cv2
import numpy as np
from cattr import unstructure
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.util import view_as_blocks
from tqdm import tqdm

from mozyq.cnst import SIZE_FOR_MATCHING
from mozyq.io import load_tiles, read_image_lab
from mozyq.mozyq_types import Mozyq


class MozyqGenerator:
    def __init__(
            self, *,
            th: int,
            tw: int,
            grid_size: int,
            paths: list[Path],
            vecs: np.ndarray,
    ):

        assert vecs.ndim == 2, f'vectors must be 2D {vecs.shape}'

        print(f'{vecs.shape=}')

        self.paths = np.array(paths)
        self.vecs = vecs
        self.th = th
        self.tw = tw
        self.grid_size = grid_size

    @classmethod
    def from_folder(cls, folder: Path, grid_size: int):
        ps = list(folder.glob('*.jpg'))

        tile = read_image_lab(ps[0])

        h, w, _ = tile.shape
        scale = max(h, w) / SIZE_FOR_MATCHING

        th = int(h / scale)
        tw = int(w / scale)
        print(f'Loading tiles at scale {h=} {w=} {th=} {tw=} {scale=}')

        tiles = load_tiles(ps, tw=tw, th=th)

        vecs = [
            tile.ravel().astype(np.float32)
            for tile in tqdm(tiles, desc='vectorizing tiles')]

        vecs = np.stack(vecs)

        return cls(paths=ps, vecs=vecs, grid_size=grid_size, tw=tw, th=th)

    def generate(self, master: np.ndarray):
        h, w, c = master.shape

        master = cv2.resize(
            master,
            (self.tw * self.grid_size, self.th * self.grid_size),
            interpolation=cv2.INTER_AREA)

        assert c == 3, 'master image must be LAB'
        assert h % 2 == 0, 'master image height must be even'
        assert w % 2 == 0, 'master image width must be even'
        assert master.size <= self.vecs.size, f'master {master.shape} vecs {self.vecs.shape}'

        master = master.astype(np.float32)

        targets = view_as_blocks(
            master,
            block_shape=(self.th, self.tw, 3)).reshape(-1, self.th * self.tw * 3)

        d = cdist(self.vecs, targets, metric='euclidean')

        rid, cid = linear_sum_assignment(d)

        # Sort indices
        return self.paths[rid][np.argsort(cid)]


def gen_mzq_json(
        *,
        master: Path,
        tile_folder: Path,
        grid_size: int,
        output_json: Path,
        max_transitions: int):

    gen = MozyqGenerator.from_folder(
        tile_folder,
        grid_size=grid_size)

    masters = set()
    mzqs: list[Mozyq] = []
    for _ in tqdm(range(max_transitions)):
        # SHOULD RARELY HAPPEN
        if master in masters:
            print(f'Master {master} already used, stopping generation')
            mzqs.pop()
            break

        masters.add(master)

        # GENERATE
        paths = gen.generate(read_image_lab(master))

        # CHOOSE STARTING POINT
        for _ in range(3):
            start = random.randint(0, len(paths) - 1)
            if paths[start] not in masters:
                break

        mzqs.append(
            Mozyq(
                master=master,
                tiles=paths.tolist(),
                start=start))

        master = paths[start]

    # WRITE JSON
    with output_json.open('w') as f:
        json.dump(unstructure(mzqs[::-1]), f)

    print(f'Wrote Mozyq JSON to {output_json}')


def gen_full_json(
        *,
        tile_folder: Path,
        grid_size: int,
        output_json: Path):

    gen = MozyqGenerator.from_folder(
        tile_folder,
        grid_size=grid_size)

    tiles = {
        str(p.stem): [
            str(p.stem)
            for p in gen.generate(read_image_lab(p))]

        for p in tqdm(gen.paths, desc='Generating full JSON')
    }

    # WRITE JSON
    with output_json.open('w') as f:
        json.dump(tiles, f)


if __name__ == '__main__':
    master = Path('./normalized/0000.jpg')
    # gen_mzq_json(
    #     master=master,
    #     tile_folder=master.parent,
    #     grid_size=9,
    #     max_transitions=2,
    #     output_json=Path('./mzq.json')
    # )

    gen_full_json(
        tile_folder=Path('./normalized'),
        grid_size=16,
        output_json=Path('./output.json'))
