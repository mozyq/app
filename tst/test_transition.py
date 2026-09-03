import numpy as np

from mozyq.transition import _gen_transition, _transition


def _make_transition(fpt):
    return _gen_transition(fpt=fpt, sx=0.0, sy=0.0, end_scale=1 / 3)


def _make_inputs():
    master = np.zeros((6, 6, 3), dtype=float)
    grid = np.zeros((18, 18, 3), dtype=float)
    return master, grid


def test_transition_frame_count_includes_crossfade():
    master, grid = _make_inputs()
    t = _make_transition(fpt=5)

    frames = list(_transition(master, grid, t, crossfade_frames=10))

    assert len(frames) == 5 + 10


def test_transition_crossfade_defaults_to_slower_60():
    master, grid = _make_inputs()
    t = _make_transition(fpt=5)

    frames = list(_transition(master, grid, t))

    assert len(frames) == 5 + 60
