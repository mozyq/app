from pathlib import Path
from typing import Any, Iterable, Literal, Tuple

from attrs import frozen

Images = Iterable[Tuple[Any, str]]


@frozen
class Mozyq:
    master: Path
    tiles: list[Path]
    start: int


Preset = Literal[
    'ultrafast',
    'superfast',
    'veryfast',
    'faster',
    'fast',
    'medium',
    'slow',
    'slower',
    'veryslow']
