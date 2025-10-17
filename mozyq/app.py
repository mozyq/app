from pathlib import Path
from typing import Annotated, get_args

import typer

from mozyq.mozyq_types import Preset

app = typer.Typer()


def path_completion(incomplete: str):
    folder = Path(incomplete).parent

    return [
        f'{folder}/{file}'
        for file in folder.iterdir()]


def preset_completion():
    return get_args(Preset)


@app.command()
def mzq(
        seed: Annotated[
            Path,
            typer.Argument(autocompletion=path_completion)],

        width: Annotated[
            int,
            typer.Option(
                help='The output width (overrides resolution for rectangular videos)')] = 630,

        height: Annotated[
            int,
            typer.Option(
                help='The output height (overrides resolution for rectangular videos)')] = 630,

        num_tiles: Annotated[
            int,
            typer.Option(
                help='The number of tiles in the grid. Must be odd and must divide evenly into the video dimensions.')] = 21,

        max_transitions: int = 10,
        output_json: Path = Path('mzq.json')
):
    '''
    Create a video from a seed image. The seed image is assumed to be in a folder with other photos.
    Usually you should have at least 1,000 photos of size at least 630x630 pixels.
    For rectangular videos, specify both width and height, or just resolution for square videos.
    '''
    from mozyq.mzq import gen_mzq_json

    # Determine final dimensions
    gen_mzq_json(
        seed=seed,
        tile_folder=seed.parent,
        master_width=width,
        master_height=height,
        tile_width=tile_width,
        max_transitions=max_transitions,
        output_json=output_json,
        grid_shape=grid_shape)


def main():
    app()
