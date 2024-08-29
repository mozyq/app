from typing import Any, Iterable, Literal, Tuple

Images = Iterable[Tuple[Any, str]]


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
