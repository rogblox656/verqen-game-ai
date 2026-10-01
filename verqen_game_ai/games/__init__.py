"""Game registry: name -> (formatter, default actions, question template).

Each game module exposes three objects and nothing else:

* ``format_state(board_state) -> str``: pure-Python board to ``<STATE>`` text.
* ``DEFAULT_ACTIONS``: actions used when the caller passes none.
* ``QUESTION``: the game-specific decision question asked of VerQen.

Register your own game with :func:`register_game`; no engine, pygame or
gymnasium import is needed to use the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import chess, game_2048, pacman, snake, tictactoe, tetris, tower_defense

__all__ = ["GameSpec", "available_games", "get_game", "register_game"]


@dataclass(frozen=True)
class GameSpec:
    """Everything ``VerQenAgent`` needs to play one game."""

    name: str
    formatter: Callable
    default_actions: tuple[str, ...]
    question: str


_REGISTRY: dict[str, GameSpec] = {}


def register_game(
    name: str,
    formatter: Callable,
    default_actions,
    question: str,
    *,
    replace: bool = False,
) -> GameSpec:
    """Register a game. Raises ``ValueError`` on duplicates unless *replace*."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("game name must be a non-empty string")
    if key in _REGISTRY and not replace:
        raise ValueError(f"game {key!r} is already registered (pass replace=True to override)")
    actions = tuple(str(a) for a in default_actions)
    if not actions:
        raise ValueError("default_actions must not be empty")
    spec = GameSpec(name=key, formatter=formatter, default_actions=actions, question=str(question))
    _REGISTRY[key] = spec
    return spec


def get_game(name: str) -> GameSpec:
    """Look up a game spec by name (case-insensitive)."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown game {name!r}. Available games: {', '.join(available_games())}")
    return _REGISTRY[key]


def available_games() -> list[str]:
    """Sorted list of registered game names."""
    return sorted(_REGISTRY)


for _module, _name in (
    (tetris, "tetris"),
    (snake, "snake"),
    (tictactoe, "tictactoe"),
    (chess, "chess"),
    (pacman, "pacman"),
    (game_2048, "2048"),
    (tower_defense, "tower_defense"),
):
    register_game(_name, _module.format_state, _module.DEFAULT_ACTIONS, _module.QUESTION)
del _module, _name
