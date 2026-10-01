"""Overlay tests: import-safe without pygame, helpful errors when called without it."""

from __future__ import annotations

import importlib.util

import pytest

from verqen_game_ai import overlay

HAS_PYGAME = importlib.util.find_spec("pygame") is not None


def test_module_imports_without_pygame():
    assert callable(overlay.draw_probability_bars)
    assert callable(overlay.create_overlay)
    assert callable(overlay._require_pygame)


def test_bars_normalization_is_pure():
    # no pygame needed: the decision -> [(action, prob)] step is pure
    assert overlay._bars({"a": 0.2, "b": 0.8}) == [("b", 0.8), ("a", 0.2)]
    assert overlay._bars(None) == []


@pytest.mark.skipif(HAS_PYGAME, reason="pygame installed")
def test_draw_raises_helpful_error_without_pygame():
    with pytest.raises(ImportError, match="pygame is required") as exc:
        overlay.draw_probability_bars(None, {"a": 1.0})
    assert "verqen-game-ai[overlay]" in str(exc.value)


@pytest.mark.skipif(HAS_PYGAME, reason="pygame installed")
def test_create_overlay_raises_helpful_error_without_pygame():
    with pytest.raises(ImportError, match="verqen-game-ai\\[overlay\\]"):
        overlay.create_overlay({"a": 1.0})
