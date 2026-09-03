"""Rasterize Noto Color Emoji SVGs into mosaic tiles that `mzq` can consume.

Source is the vector `svg/` directory of googlefonts/noto-emoji (sparse-cloned
to `demo/noto-emoji/` by `task emoji:svg`), rasterized at 2x and downsampled
for crisp edges -- no upscaled-bitmap pixelation.

Tile style: the emoji, bbox-normalised to a fixed size, sits on a flat
background whose hue comes from the emoji's dominant colour but whose
lightness is pushed the opposite way, so the emoji always has contrast and
the tile still carries a colour for `mzq` to match the master against.

Candidates are the single-code-point emoji in the curated RANGES plus the ZWJ
sequences (astronaut, firefighter, rainbow flag, ...); the ~2000 skin-tone
variants of those sequences are skipped as near-duplicates.

Two filters keep junk out of the set:
  * automatic -- empty glyphs and busy "scene" emoji (cityscape, map, ...)
  * demo/emoji_blocklist.txt -- hand-picked code points that just look muddy

Output `<repo>/emojis/`:
    0000_master.jpg   the seed image (an emoji on white, or EMOJI_MASTER's image)
    <hex>.jpg         one file per emoji, named by its code point(s), '_'-joined
    _filtered.txt     every dropped glyph and why (for review)

The output is uniform, so `mzq normalize` is not needed. Full demo: `task emoji`
(render + review tiles first -- see `task emoji:tiles`).
"""
from __future__ import annotations

import io
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SVG_DIR = REPO / "demo" / "noto-emoji" / "svg"
BLOCKLIST = REPO / "demo" / "emoji_blocklist.txt"
OUT = REPO / "emojis"

SIZE = int(os.environ.get("EMOJI_SIZE", "512"))       # px, square tile
N_TILES = int(os.environ.get("EMOJI_COUNT", "2000"))  # cap; the pool is ~1500
SS = int(os.environ.get("EMOJI_SUPERSAMPLE", "2"))    # raster oversample
EMOJI_FRAC = float(os.environ.get("EMOJI_FRAC", "0.82"))  # emoji size in the tile
STYLE = os.environ.get("EMOJI_STYLE", "contrast")     # "contrast" | "white"

# The seed image. Either an emoji (hex code point / "U+..." / a literal emoji
# char) rendered flat on white, or a path to an image file that gets
# centre-cropped square and resized to SIZE. Default: 🎨 artist palette.
MASTER = os.environ.get("EMOJI_MASTER", "1f3a8").strip()

STAT_PX = 96  # colour stats run on a small thumbnail; keep thresholds in sync

SKIN = {0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF}  # Fitzpatrick modifiers
ZWJ = 0x200D

RANGES = [
    (0x1F300, 0x1F5FF),  # misc symbols & pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport & map
    (0x1F900, 0x1F9FF),  # supplemental symbols & pictographs
    (0x1FA70, 0x1FAFF),  # symbols & pictographs extended-A
    (0x2600, 0x27BF),    # misc symbols & dingbats
]


class Stat(NamedTuple):
    opaque: float      # fraction of the frame the emoji covers
    dom: np.ndarray    # dominant opaque colour (RGB float)
    lum: float         # mean opaque luminance, 0..1
    ncols: int         # distinct coarse colour buckets
    share: float       # fraction of opaque pixels in the dominant bucket


def svg_path(cp: int) -> Path:
    return SVG_DIR / f"emoji_u{cp:04x}.svg"


def _cps(stem: str) -> list[int]:
    # "emoji_u1f468_200d_1f680" -> [0x1f468, 0x200d, 0x1f680]
    return [int(t, 16) for t in stem.removeprefix("emoji_u").split("_")]


def _emoji_cp(tok: str) -> int:
    # "1f3a8" / "U+1F3A8" / "0x1F3A8" / "🎨" -> code point
    tok = tok[2:] if tok.lower().startswith(("u+", "0x")) else tok
    try:
        return int(tok, 16)
    except ValueError:
        return ord(tok[0])


# Resolved once: an image file to use verbatim, else the emoji code point.
MASTER_SRC = Path(MASTER) if Path(MASTER).is_file() else None
MASTER_CP = None if MASTER_SRC else _emoji_cp(MASTER)


def load_blocklist() -> set[int]:
    blocked: set[int] = set()
    if not BLOCKLIST.exists():
        return blocked
    for line in BLOCKLIST.read_text().splitlines():
        tok = line.split("#", 1)[0].strip()
        if not tok:
            continue
        if tok.lower().startswith(("u+", "0x")):
            tok = tok[2:]
        try:
            blocked.add(int(tok, 16))
        except ValueError:
            blocked.add(ord(tok[0]))  # a literal emoji character
    return blocked


def collect_candidates(blocked: set[int]) -> list[tuple[str, Path]]:
    """(name, svg) pairs: single code points in RANGES + ZWJ sequences.

    `name` is the output stem -- the '_'-joined hex code point(s). Skin-tone
    variants of the sequences are dropped as near-duplicates.
    """
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for lo, hi in RANGES:
        for cp in range(lo, hi + 1):
            if cp == MASTER_CP or cp in blocked:
                continue
            p = svg_path(cp)
            if p.exists() and (name := f"{cp:x}") not in seen:
                seen.add(name)
                out.append((name, p))

    for p in sorted(SVG_DIR.glob("emoji_u*_200d_*.svg")):
        try:
            cps = _cps(p.stem)
        except ValueError:
            continue
        if ZWJ not in cps or SKIN & set(cps) or cps[0] in blocked:
            continue
        name = p.stem.removeprefix("emoji_u")
        if name not in seen:
            seen.add(name)
            out.append((name, p))

    return out


def render_rgba(svg: int | Path, px: int) -> Image.Image:
    path = svg_path(svg) if isinstance(svg, int) else svg
    raw = subprocess.run(
        ["rsvg-convert", "-w", str(px * SS), "-h", str(px * SS), str(path)],
        check=True, capture_output=True,
    ).stdout
    return Image.open(io.BytesIO(raw)).convert("RGBA").resize((px, px), Image.Resampling.BOX)


def stats(im: Image.Image) -> Stat:
    small = np.asarray(im.resize((STAT_PX, STAT_PX), Image.Resampling.BOX))
    rgb = small[..., :3].astype(np.int16)
    m = small[..., 3] > 180
    opq = float(m.mean())
    px = rgb[m]
    if len(px) < 40:
        return Stat(opq, np.array([235.0, 235, 235]), 0.6, 0, 0.0)
    q = px // 26
    codes = q[:, 0] * 121 + q[:, 1] * 11 + q[:, 2]
    vals, counts = np.unique(codes, return_counts=True)
    top = vals[counts.argmax()]
    dom = px[codes == top].mean(0).astype(float)
    lum = float((px @ np.array([0.299, 0.587, 0.114])).mean()) / 255.0
    return Stat(opq, dom, lum, len(vals), float(counts.max()) / len(px))


def reject_reason(s: Stat) -> str | None:
    # No heuristics: bad-looking tiles are curated by hand in
    # demo/emoji_blocklist.txt after reviewing the montage from `task emoji:tiles`.
    # The only automatic skip is a glyph that rasterised to nothing.
    if s.opaque < 0.03:
        return "empty"
    return None


def _mix(x, y, t):
    return np.asarray(x, float) * (1 - t) + np.asarray(y, float) * t


def make_tile(im: Image.Image, s: Stat) -> Image.Image:
    a = np.asarray(im)
    w, h = im.size

    ys, xs = np.where(a[..., 3] > 8)
    emoji = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = EMOJI_FRAC * w / max(emoji.size)
    emoji = emoji.resize((max(1, round(emoji.width * scale)),
                          max(1, round(emoji.height * scale))),
                         Image.Resampling.LANCZOS)

    if STYLE == "white":
        bg = (255, 255, 255)
    else:
        light = _mix([20, 20, 20], [245, 245, 245], 0.15 if s.lum > 0.5 else 0.9)
        c = np.clip(_mix(_mix(s.dom, light, 0.55), [128, 128, 128], -0.15), 0, 255)
        bg = tuple(int(v) for v in c)

    tile = Image.new("RGBA", (w, h), bg + (255,))
    tile.alpha_composite(emoji, ((w - emoji.width) // 2, (h - emoji.height) // 2))
    return tile.convert("RGB")


def flat_on_white(im: Image.Image) -> Image.Image:
    a = np.asarray(im, dtype=np.float64)
    al = a[..., 3:] / 255.0
    return Image.fromarray(
        (255.0 * (1 - al) + a[..., :3] * al).astype(np.uint8), "RGB")


def write_master() -> None:
    """Save the seed image to emojis/0000_master.jpg (see MASTER)."""
    if MASTER_SRC is not None:
        im = Image.open(MASTER_SRC).convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    else:
        assert MASTER_CP is not None
        im = flat_on_white(render_rgba(MASTER_CP, SIZE))
    im.save(OUT / "0000_master.jpg", "JPEG", quality=95)


def main() -> int:
    if not SVG_DIR.is_dir():
        sys.exit(f"missing {SVG_DIR} -- run `task emoji:svg` first")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    blocked = load_blocklist()
    candidates = collect_candidates(blocked)
    random.Random(0).shuffle(candidates)  # deterministic spread across ranges

    kept = 0
    dropped: Counter[str] = Counter()
    drop_log: list[str] = []
    for name, svg in candidates:
        if kept >= N_TILES:
            break
        im = render_rgba(svg, SIZE)
        s = stats(im)
        why = reject_reason(s)
        if why:
            dropped[why] += 1
            drop_log.append(f"{name}\t{why}")
            continue
        make_tile(im, s).save(OUT / f"{name}.jpg", "JPEG", quality=95)
        kept += 1

    write_master()

    (OUT / "_filtered.txt").write_text("\n".join(drop_log) + "\n")
    m = MASTER_SRC.name if MASTER_SRC else f"U+{MASTER_CP:04X}"
    print(f"kept {kept} tiles  |  master {m}  |  blocklist {len(blocked)}  |  "
          f"auto-dropped {sum(dropped.values())} {dict(dropped)}")
    print(f"drop details -> {OUT / '_filtered.txt'}")
    return 0 if kept >= 1000 else 1


if __name__ == "__main__":
    sys.exit(main())
