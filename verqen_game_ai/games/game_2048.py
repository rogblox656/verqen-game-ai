"""2048: board formatter, default actions and question text.

The board is a 4x4 nested list, a flat list of 16 values, or a dict with
``grid`` and an optional ``score``.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = ("up", "down", "left", "right")
QUESTION = "Which direction should the 2048 board slide?"


def _row_text(row) -> str:
    return " ".join("0" if cell is None else str(cell) for cell in row)


def _rows(grid) -> list:
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("2048 state needs a non-empty 'grid' list")
    if all(isinstance(row, (list, tuple)) for row in grid):
        return list(grid)
    if len(grid) == 16:
        return [list(grid[i : i + 4]) for i in range(0, 16, 4)]
    raise ValueError("2048 board must be 4 rows or a flat list of 16 values")


def format_state(board_state) -> str:
    """``<STATE> 2048 board: <4 rows> [score: n]``."""
    if isinstance(board_state, dict):
        grid = board_state.get("grid", board_state.get("board"))
        score = board_state.get("score")
    else:
        grid, score = board_state, None
    body = f"2048 board: {'/'.join(_row_text(row) for row in _rows(grid))}"
    if score is not None:
        body += f" score: {score}"
    return state_text(body)
