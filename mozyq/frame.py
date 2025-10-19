# https://names2023war.ynet.co.il/api/people/paginate?limit=2000&offset=0&search=
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mozyq.io import write_jpeg
from mozyq.mzq import Mozyq
from mozyq.transition import mzq_transition


def frames(
        *,
        mzqs: list[Mozyq],
        fpt: int,
        out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)

    i = 0
    for mzq in mzqs:
        fs = mzq_transition(mzq, fpt=fpt)

        for frame in tqdm(fs):
            write_jpeg(
                frame.astype(np.uint8),
                out_folder / f'{i:04d}.jpg')

            i += 1
