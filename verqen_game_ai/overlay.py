"""Pygame decision overlay: draw VerQen's probability bars over your game.

Import-safe without pygame: the dependency is imported inside each function
and a helpful ``ImportError`` is raised when it is missing. Install it with::

    pip install "verqen-game-ai[overlay]"
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_BAR_BG = (32, 36, 44)
_BAR_FILL = (214, 255, 75)
_BAR_TEXT = (235, 240, 245)
_LABEL_WIDTH = 110


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for the decision overlay but is not installed. "
            'Install it with: pip install "verqen-game-ai[overlay]"'
        ) from exc
    return pygame


def _bars(decision: Any) -> list[tuple[str, float]]:
    """Normalize a GameDecision or a plain {action: prob} mapping to bars."""
    if isinstance(decision, Mapping):
        items = list(decision.items())
    else:
        items = list(getattr(decision, "alternatives", []) or [])
        if not items:
            probabilities = getattr(decision, "probabilities", None)
            items = list(probabilities.items()) if probabilities else []
    return sorted(((str(a), float(p)) for a, p in items), key=lambda kv: -kv[1])


def draw_probability_bars(surface, decision: Any, x: int = 8, y: int = 8, width: int = 240) -> int:
    """Draw one labeled probability bar per action onto *surface*.

    Accepts a :class:`~verqen_game_ai.GameDecision` (uses its top-5
    ``alternatives``) or a plain ``{action: probability}`` mapping. Returns
    the height consumed in pixels.
    """
    pygame = _require_pygame()
    bars = _bars(decision)
    if not bars:
        return 0
    row_height = 20
    font = pygame.font.SysFont("consolas", 14)
    max_p = max((p for _, p in bars), default=0.0) or 1.0
    for index, (action, prob) in enumerate(bars):
        top = y + index * row_height
        label = font.render(action[:14], True, _BAR_TEXT)
        track = pygame.Rect(x + _LABEL_WIDTH, top + 3, width - _LABEL_WIDTH, row_height - 8)
        pygame.draw.rect(surface, _BAR_BG, track)
        fill = track.copy()
        fill.width = max(1, int(track.width * min(1.0, prob / max_p)))
        pygame.draw.rect(surface, _BAR_FILL, fill)
        value = font.render(f"{prob:.2f}", True, _BAR_TEXT)
        surface.blit(label, (x, top + 2))
        surface.blit(value, (track.right + 6, top + 2))
    return len(bars) * row_height


def create_overlay(decision: Any, width: int = 300) -> "object":
    """Render the overlay into a fresh translucent ``pygame.Surface``.

    Blit it onto your game surface each frame::

        surface.blit(create_overlay(decision), (10, 10))
    """
    pygame = _require_pygame()
    bars = _bars(decision)
    height = max(1, len(bars)) * 20 + 8
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((10, 12, 16, 200))
    draw_probability_bars(overlay, bars, x=4, y=4, width=width - 8)
    return overlay
