"""Chess: board formatter, default actions and question text.

The state is a FEN string, or a dict with a ``fen`` key. This formatter is
pure Python: pass UCI moves (for example from ``python-chess``'s
``legal_moves``) as the available actions at call time. ``DEFAULT_ACTIONS``
is only a fallback opening set for callers that pass no actions.
"""

from __future__ import annotations

from ..state_formatter import state_text

DEFAULT_ACTIONS = ("e2e4", "d2d4", "g1f3", "b1c3", "f1c4")
QUESTION = "What is the best move in this chess position?"


def format_state(board_state) -> str:
    """``<STATE> chess fen: <FEN>``."""
    fen = board_state.get("fen") if isinstance(board_state, dict) else board_state
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("chess state must be a FEN string or a dict with a 'fen' key")
    return state_text(f"chess fen: {fen.strip()}")
