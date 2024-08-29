import json
from math import sqrt
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from attr import dataclass
from attrs import frozen
from cattrs import unstructure
from scipy.optimize import linear_sum_assignment as lsa
from torch import Tensor
from torch.nn import Unfold
from tqdm import tqdm

from mozyq.io import load_img_any_size, load_tiles


@frozen
class Mozyq:
    uuid: str
    master: Path
    tiles: list[Path]

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __eq__(self, other) -> bool:
        return self.uuid == other.uuid

    @property
    def nrow(self):
        nrow = int(sqrt(len(self.tiles)))
        assert nrow ** 2 == len(self.tiles), \
            f'len(tiles) must be a perfect square {len(self.tiles)}'

        return nrow

    @property
    def grid(self):
        n = int(np.sqrt(len(self.tiles)))
        assert n ** 2 == len(self.tiles), 'tiles must be square'
        return np.array(self.tiles).reshape(n, n)


class MozyqGenerator:
    def __init__(
            self,
            *,
            paths: list[Path],
            vecs: Tensor,
            tile_size: int):

        assert vecs.dim() == 2, f'vectors must be 2D {vecs.shape}'

        _, s = vecs.shape

        assert s == tile_size ** 2 * 3, \
            f'vectors must be of size {tile_size ** 2 * 3}'

        self.paths = np.array(paths)
        self.vecs = vecs
        self.tile_size = tile_size

    @classmethod
    def from_folder(cls, folder: Path, *, tile_size: int):
        ps = sorted(list(folder.glob('*.jpg')))

        tiles = load_tiles(
            tqdm(ps, desc='reading tiles'),
            tile_size)

        vecs = [
            tile.ravel().to(torch.float32)
            for tile in tqdm(tiles, desc='vectorizing tiles')]

        vecs = torch.stack(vecs)

        return cls(paths=ps, vecs=vecs, tile_size=tile_size)

    def generate(self, master: Tensor) -> np.ndarray:
        c, h, w = master.shape

        assert c == 3, 'master image must be RGB'
        assert h == w, 'master image must be square'
        assert h % self.tile_size == 0, \
            f'master image must be divisible by tile_size {master.shape}'

        assert h % 2 == 0, 'master image must be even'
        assert master.nelement() <= self.vecs.nelement(), 'master image too large'

        unfold = Unfold(
            kernel_size=self.tile_size,
            stride=self.tile_size)

        master = master.to(torch.float32)

        targets = unfold(master.unsqueeze(0)).squeeze(0).T

        d = torch.cdist(self.vecs[None], targets[None]).squeeze()
        rid, cid = lsa(d.numpy())

        _, ids = torch.sort(torch.asarray(cid))
        return self.paths[rid][ids]


@dataclass
class Video:
    master_size: int
    mozyqs: list[Mozyq]


def save_video_json(
        *, seed: Path,
        tile_folder: Path,
        master_size: int,
        tile_size: int,
        num_transitions: int,
        video_json: Path):

    assert master_size % 2 == 0, 'target_size must be even'
    assert master_size % tile_size == 0, 'target_size must be divisible by tile_size'

    gen = MozyqGenerator.from_folder(
        tile_folder,
        tile_size=tile_size)

    mozyqs: list[Mozyq] = []

    for _ in range(num_transitions):
        tiles = gen.generate(load_img_any_size(seed, master_size))

        mozyq = Mozyq(
            uuid=uuid4().hex,
            master=seed,
            tiles=tiles.tolist())

        assert len(tiles) % 2 == 1, 'len(tiles) must be odd'

        seed = tiles[len(tiles) // 2]

        mozyqs.append(mozyq)

    video = Video(
        master_size=master_size,
        mozyqs=mozyqs[::-1])

    with open(video_json, 'w') as f:
        video = unstructure(video)
        print(json.dumps(video), file=f)
