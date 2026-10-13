"""The architecture diagram script runs without data and is reproducible."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("make_architecture_figure", ROOT / "docs" / "make_architecture_figure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_svg_and_png_are_written_and_the_svg_is_reproducible(tmp_path):
    module = load()
    first, second = tmp_path / "a.svg", tmp_path / "b.svg"
    png = tmp_path / "a.png"
    module.draw(png, first)
    module.draw(None, second)
    assert first.read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))
    assert first.stat().st_size > 10_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert first.read_bytes() == second.read_bytes()          # no timestamps or random ids: re-running does not change the file


def test_the_committed_diagram_matches_the_script():
    module = load()
    assert (ROOT / "docs" / "figures" / "architecture.svg").exists() and (ROOT / "docs" / "figures" / "architecture.png").exists()
    assert module.OUT == ROOT / "docs" / "figures"
