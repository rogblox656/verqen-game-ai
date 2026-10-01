"""Convert game board states into VerQen ``<STATE>`` text.

VerQen reads a single text field per decision. Every formatter in this
package turns a native Python board representation (dict, list or string)
into a compact, deterministic ``"<STATE> ..."`` line.

``generic_format`` handles any dict/list/str without game-specific rules;
``get_formatter`` resolves the per-game formatter from the registry in
:mod:`verqen_game_ai.games`. Both are pure Python: no pygame, gymnasium,
python-chess or verqen import happens here.
"""

from __future__ import annotations

from typing import Any, Callable

STATE_PREFIX = "<STATE>"


def state_text(body: str) -> str:
    """Wrap *body* in the ``<STATE>`` prefix VerQen expects."""
    return f"{STATE_PREFIX} {body.strip()}"


def _scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k}={_render(v)}" for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(row, (list, tuple)) for row in value):
            return "/".join(" ".join(_scalar(cell) for cell in row) for row in value)
        return " ".join(_scalar(item) for item in value)
    return _scalar(value)


def generic_format(board_state: Any) -> str:
    """Canonical ``<STATE>`` text for any dict/list/str board representation.

    Scalars render as themselves, lists of scalars as space-separated values,
    nested lists as slash-separated rows, and dicts as sorted ``key=value``
    pairs (recursively).
    """
    if isinstance(board_state, str):
        return state_text(board_state)
    return state_text(_render(board_state))


def get_formatter(game: str) -> Callable:
    """Return the formatter callable registered for *game*.

    Raises ``ValueError`` for unknown games, listing the available ones.
    """
    from .games import get_game  # local import: games registry imports formatters

    return get_game(game).formatter
