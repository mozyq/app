from pathlib import Path

import yaml

TASKFILE = Path(__file__).parent.parent / "Taskfile.yml"


def _publish_cmds():
    taskfile = yaml.safe_load(TASKFILE.read_text())
    return taskfile["tasks"]["publish"]["cmds"]


def test_publish_tests_before_bumping_version():
    cmds = _publish_cmds()

    test_step = next(i for i, c in enumerate(cmds) if "pytest" in c)
    bump_step = next(i for i, c in enumerate(cmds) if "uv version --bump" in c)

    assert test_step < bump_step


def test_publish_checks_clean_worktree_before_bumping_version():
    cmds = _publish_cmds()

    clean_check_step = next(i for i, c in enumerate(cmds) if "git status --porcelain" in c)
    bump_step = next(i for i, c in enumerate(cmds) if "uv version --bump" in c)

    assert clean_check_step < bump_step


def test_publish_bumps_version_before_publishing():
    cmds = _publish_cmds()

    bump_step = next(i for i, c in enumerate(cmds) if "uv version --bump" in c)
    publish_step = next(i for i, c in enumerate(cmds) if c == "uv publish")

    assert bump_step < publish_step


def test_publish_does_not_use_poetry():
    cmds = _publish_cmds()

    assert not any("poetry" in c for c in cmds)
