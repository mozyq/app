from pathlib import Path
from typing import Annotated, cast, get_args

import typer

from mozyq.types import Preset

app = typer.Typer()


def path_completion(incomplete: str):
    folder = Path(incomplete).parent

    return [
        f'{folder}/{file}'
        for file in folder.iterdir()]


def preset_completion():
    return get_args(Preset)


@ app.command()
def mzq(
        seed: Annotated[
            Path,
            typer.Argument(autocompletion=path_completion)],

        video_mp4: Annotated[
            Path,
            typer.Option(
                autocompletion=path_completion,
                help='The output video file.')] = Path('video.mp4'),

        resolution: Annotated[
            int,
            typer.Option(
                help='The output resolution')] = 630,

        tile_size: Annotated[
            int,
            typer.Option(
                help='The size of each tile. The resolution must be devisable by the tile size. And there should be an odd number of tiles.')] = 30,

        transitions: int = 10,
        fps: int = 180,
        crf: int = 18,
        preset: Annotated[
            str,
            typer.Option(
                autocompletion=preset_completion,
                help='The x264 preset to use.')] = 'medium'):
    '''
    Create a video from a seed image. The seed image is assumed to be in a folder with other photos.
    Usually you should have at least 1,000 of size at least 630x630 pixels.
    '''
    from mozyq.mzq import save_video_json as svj
    from mozyq.vid import build_video as bv

    video_mp4.parent.mkdir(parents=True, exist_ok=True)
    video_json = video_mp4.with_suffix('.json')

    svj(
        seed=seed,
        tile_folder=seed.parent,
        master_size=resolution,
        tile_size=tile_size,
        num_transitions=transitions,
        video_json=video_json)

    bv(
        video_json=video_json,
        video_mp4=video_mp4,
        steps_per_transition=fps,
        crf=crf,
        preset=cast(Preset, preset))


def main():
    app()
