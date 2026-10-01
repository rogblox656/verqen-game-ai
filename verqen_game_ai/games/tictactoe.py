"""Tic-tac-toe: board formatter, default actions and question text.

The board is a 9-cell row-major string (``X`` / ``O`` / ``.``), a 3x3 list,
or a dict with ``board`` and an optional ``to_move`` side.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = (
    "top_left",
    "top_center",
    "top_right",
    "mid_left",
    "center",
    "mid_right",
    "bot_left",
    "bot_center",
    "bot_right",
)
QUESTION = "What is the best square to play in this tic-tac-toe position?"


def _flatten(raw) -> str:
    if isinstance(raw, str):
        cells = raw
    elif isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise ValueError(f"tictactoe board must have 3 rows, got {len(raw)}")
        flat: list[str] = []
        for row in raw:
            row_cells = list(row) if isinstance(row, (list, tuple, str)) else None
            if row_cells is None:
                raise ValueError("tictactoe board rows must be strings or sequences")
            if len(row_cells) != 3:
                raise ValueError(f"tictactoe rows must have 3 cells, got {len(row_cells)}")
            flat.extend("." if cell is None else str(cell) for cell in row_cells)
        cells = "".join(flat)
    else:
        raise TypeError("tictactoe board must be a 9-char string, a 3x3 list or a dict")
    if len(cells) != 9:
        raise ValueError(f"tictactoe board must have 9 cells, got {len(cells)}")
    cells = cells.upper()
    invalid = set(cells) - set("XO.")
    if invalid:
        raise ValueError(f"tictactoe board has invalid cells: {sorted(invalid)}")
    return cells


def format_state(board_state) -> str:
    """``<STATE> tictactoe board: <9 cells> to_move: X|O|-``."""
    if isinstance(board_state, dict):
        raw = board_state.get("board", board_state.get("grid"))
        if raw is None:
            raise ValueError("tictactoe dict state needs a 'board' key")
        to_move = board_state.get("to_move")
    else:
        raw = board_state
        to_move = None
    cells = _flatten(raw)
    side = str(to_move).upper() if str(to_move).upper() in {"X", "O"} else "-"
    return state_text(f"tictactoe board: {cells} to_move: {side}")
