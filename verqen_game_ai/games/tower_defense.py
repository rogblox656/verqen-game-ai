"""Tower defense: state formatter, default actions and question text.

The state is a dict of counters: ``wave``, ``gold``, ``lives``, ``enemies``
(int or list of enemy entries) and ``towers`` (int or list of towers).
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = ("build_archer", "build_cannon", "build_slow", "upgrade", "save_gold")
QUESTION = "What is the best action in this tower defense state?"


def _count(value) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value)


def format_state(board_state) -> str:
    """``<STATE> tower_defense wave: n gold: n lives: n enemies: n towers: n``."""
    if not isinstance(board_state, dict):
        raise ValueError("tower_defense state must be a dict of counters")
    wave = _count(board_state.get("wave", 0))
    gold = _count(board_state.get("gold", 0))
    lives = _count(board_state.get("lives", 0))
    enemies = _count(board_state.get("enemies", 0))
    towers = _count(board_state.get("towers", 0))
    return state_text(
        f"tower_defense wave: {wave} gold: {gold} lives: {lives} "
        f"enemies: {enemies} towers: {towers}"
    )
