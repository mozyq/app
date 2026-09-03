# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mozyq` is a Python CLI (`mzq`) that builds mosaic videos from photo collections. Source is in `mozyq/`, entry point `mozyq.app:main`.

## Tooling

- **Use `uv`, not poetry.** Both `uv.lock` and `poetry.lock` exist and `Taskfile.yml` still calls poetry, but `uv` is canonical: `uv sync`, `uv run pytest`, `uv run pyright`.
- Python 3.10–3.12 (`requires-python = ">=3.10,<3.13"`, `.python-version` pins 3.10).
- `ffmpeg` must be installed on the system — it is an external runtime dependency, not a pip package.

## Tests

- Tests live in `tst/` (intentional, not `tests/`). Run with `uv run pytest`.
- Single test: `uv run pytest tst/test_builder.py -k test_name`.

## Type checking

`uv run pyright` (configured only via dev-dependency; no pyrightconfig).

## CLI options

`mozyq/app.py` is the source of truth for command names and options. The README is out of date (e.g. it documents `--num-tiles`/`--width`/`--height`; the code uses `--grid-size` and has an undocumented `json-full` command). Fix README drift opportunistically.

## Releasing

`task publish` is the whole release process (`poetry version patch` → `poetry publish --build` → git tag → commit → push). The version lives in **one place**, `pyproject.toml`; `mozyq/__init__.py` `__version__` reads it back via `importlib.metadata`, so never hardcode a version there.
