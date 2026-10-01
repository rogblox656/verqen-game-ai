"""Tetris: board formatter, default actions and question text.

The board is a dict with ``grid`` (list of rows, top to bottom), optional
``piece``, ``next`` and ``score``, or a bare grid list.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = (
    "move_left",
    "move_right",
    "rotate_left",
    "rotate_right",
    "drop",
    "hold",
)
QUESTION = "What is the best move for this Tetris board?"


def _row_text(row) -> str:
    if isinstance(row, str):
        return row
    return "".join("." if cell is None else str(cell) for cell in row)


def format_state(board_state) -> str:
    """``<STATE> tetris grid: <rows> piece: p next: p score: n``."""
    if isinstance(board_state, dict):
        grid = board_state.get("grid", board_state.get("board"))
        piece = board_state.get("piece", "-")
        next_piece = board_state.get("next", board_state.get("next_piece", "-"))
        score = board_state.get("score", 0)
    else:
        grid, piece, next_piece, score = board_state, "-", "-", 0
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("tetris state needs a non-empty 'grid' list")
    rows = [_row_text(row) for row in grid]
    return state_text(
        f"tetris grid: {'/'.join(rows)} piece: {piece} next: {next_piece} score: {score}"
    )
