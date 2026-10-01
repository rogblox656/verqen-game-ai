"""Snake: board formatter, default actions and question text.

The board is a dict with ``grid`` (list of rows), optional ``head``,
``food``, ``direction`` and ``length``, or a bare grid list.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = ("up", "down", "left", "right")
QUESTION = "Which direction should the snake move next?"


def _row_text(row) -> str:
    if isinstance(row, str):
        return row
    return "".join("." if cell is None else str(cell) for cell in row)


def _point(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"r{value[0]}c{value[1]}"
    if isinstance(value, str):
        return value
    return "-"


def format_state(board_state) -> str:
    """``<STATE> snake grid: <rows> head: rXcY food: rXcY direction: d length: n``."""
    if isinstance(board_state, dict):
        grid = board_state.get("grid", board_state.get("board"))
        head = _point(board_state.get("head", "-"))
        food = _point(board_state.get("food", "-"))
        direction = str(board_state.get("direction", "-"))
        length = board_state.get("length", "-")
    else:
        grid = board_state
        head = food = direction = length = "-"
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("snake state needs a non-empty 'grid' list")
    rows = [_row_text(row) for row in grid]
    return state_text(
        f"snake grid: {'/'.join(rows)} head: {head} food: {food} "
        f"direction: {direction} length: {length}"
    )
