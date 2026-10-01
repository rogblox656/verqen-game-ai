"""Pac-Man: maze formatter, default actions and question text.

The state is a dict with ``maze`` (list of rows), optional ``pacman`` and
``ghosts`` positions (``[row, col]``), and ``pellets`` remaining, or a bare
maze list.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = ("up", "down", "left", "right", "stay")
QUESTION = "Which way should Pac-Man move?"


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
    """``<STATE> pacman maze: <rows> pacman: rXcY ghosts: n pellets: n``."""
    if isinstance(board_state, dict):
        maze = board_state.get("maze", board_state.get("grid"))
        pacman = _point(board_state.get("pacman", "-"))
        ghosts = board_state.get("ghosts", 0)
        pellets = board_state.get("pellets", 0)
    else:
        maze, pacman, ghosts, pellets = board_state, "-", 0, 0
    if isinstance(ghosts, (list, tuple)):
        ghosts = len(ghosts)
    if not isinstance(maze, (list, tuple)) or not maze:
        raise ValueError("pacman state needs a non-empty 'maze' list")
    rows = [_row_text(row) for row in maze]
    return state_text(
        f"pacman maze: {'/'.join(rows)} pacman: {pacman} ghosts: {ghosts} pellets: {pellets}"
    )
