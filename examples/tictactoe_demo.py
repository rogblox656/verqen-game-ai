"""Play a few tic-tac-toe moves with VerQenAgent.

Uses a live VerQen checkpoint when ``VERQEN_CHECKPOINT`` is set; otherwise it
falls back to a tiny built-in rule engine (win, block, else center/corner) so
the demo runs on any machine with no GPU. Never imported by the test suite.

Run::

    python examples/tictactoe_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without pip install

from verqen_game_ai import VerQenAgent  # noqa: E402
from verqen_game_ai.games.tictactoe import DEFAULT_ACTIONS  # noqa: E402

CELLS = list(DEFAULT_ACTIONS)
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]
CENTER = 4
CORNERS = (0, 2, 6, 8)


class RuleEngine:
    """Tiny heuristic stand-in with the VerQen ``decide`` signature.

    Picks an immediate win, else blocks the opponent's win, else center,
    else a corner. For noul (yes/no) questions it reports whether the side
    to move has an immediate winning square.
    """

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        body = state.split("<STATE> ", 1)[1]
        board = body.split("board: ", 1)[1].split()[0]
        to_move = body.split("to_move: ", 1)[1].strip() or "X"
        options = [str(o) for o in (options or CELLS)]

        if question_type == "noul":
            value = _find_win(board, to_move) is not None
            chosen = "yes" if value else "no"
            return {
                "selected": chosen,
                "probabilities": {"yes": 0.9 if value else 0.1,
                                  "no": 0.1 if value else 0.9},
                "p_correct": 0.9,
                "confidence": 0.9,
                "margin": 0.8,
                "should_escalate": False,
                "prediction_set": [chosen],
                "value": value,
                "p_true": 0.9 if value else 0.1,
            }

        move = _find_win(board, to_move)
        if move is None:
            move = _find_win(board, _opponent(to_move))  # block
        if move is None and board[CENTER] == ".":
            move = CENTER
        if move is None:
            move = next((i for i in CORNERS if board[i] == "."), None)
        if move is None:
            move = next((i for i, ch in enumerate(board) if ch == "."), None)
        chosen = CELLS[move] if move is not None else options[0]
        probs = {o: (0.9 if o == chosen else 0.1 / max(1, len(options) - 1))
                 for o in options}
        return {
            "selected": chosen,
            "probabilities": probs,
            "p_correct": 0.9,
            "confidence": 0.9,
            "margin": 0.8,
            "should_escalate": False,
            "prediction_set": [chosen],
        }


def _opponent(side: str) -> str:
    return "O" if side == "X" else "X"


def _find_win(board: str, side: str) -> int | None:
    """Empty cell index that completes a line for *side*, else None."""
    for a, b, c in LINES:
        cells = (board[a], board[b], board[c])
        if cells.count(side) == 2 and cells.count(".") == 1:
            return (a, b, c)[cells.index(".")]
    return None


def _winner(board: str) -> str | None:
    for a, b, c in LINES:
        if board[a] == board[b] == board[c] and board[a] in "XO":
            return board[a]
    return None


def main() -> int:
    if os.environ.get("VERQEN_CHECKPOINT"):
        agent = VerQenAgent.from_pretrained(game="tictactoe")
        print("engine: live VerQen checkpoint")
    else:
        agent = VerQenAgent(game="tictactoe", engine=RuleEngine())
        print("engine: built-in rule fallback (set VERQEN_CHECKPOINT for live play)")

    board = "." * 9
    to_move = "X"
    moves = 0

    # One advisory question up front (noul mode).
    opening_advice = agent.check(
        "Can the player to move win immediately?", {"board": board, "to_move": to_move}
    )
    print(f"opening: player to move has an immediate win? {opening_advice}")

    while moves < 9:
        decision = agent.decide({"board": board, "to_move": to_move}, CELLS)
        index = CELLS.index(decision.action)
        print(
            f"{to_move} plays {decision.action:11s} "
            f"conf={decision.confidence:.2f} set={decision.conformal_actions} "
            f"({decision.latency_ms:.1f} ms)"
        )
        board = board[:index] + to_move + board[index + 1 :]
        moves += 1
        if _winner(board):
            break
        to_move = _opponent(to_move)

    result = _winner(board)
    print()
    print(f"final board ({'winner: ' + result if result else 'draw'}):")
    for row in range(3):
        print("  " + " | ".join(board[row * 3 + col] for col in range(3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
