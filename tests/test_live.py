"""Live tests against a real VerQen checkpoint.

Skipped unless ``VERQEN_CHECKPOINT`` is set. Run with::

    pytest -m live
"""

from __future__ import annotations

import os
import time

import pytest

from verqen_game_ai import VerQenAgent
from verqen_game_ai.games.tictactoe import DEFAULT_ACTIONS

CELLS = list(DEFAULT_ACTIONS)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("VERQEN_CHECKPOINT"),
        reason="VERQEN_CHECKPOINT not set",
    ),
]


@pytest.fixture(scope="module")
def live_agent():
    return VerQenAgent.from_pretrained(game="tictactoe")


def test_decide_returns_legal_action(live_agent):
    decision = live_agent.decide({"board": "XX.O.....", "to_move": "O"}, CELLS)
    assert decision.action in CELLS
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.conformal_actions
    assert decision.latency_ms > 0
    assert len(decision.alternatives) >= 1


def test_check_returns_bool(live_agent):
    answer = live_agent.check("Does X have two in a row?", "XX......O")
    assert isinstance(answer, bool)


def test_latency_budget_eager(live_agent):
    """Warm call budget: eager mode is ~68ms on an RTX 5060 Ti; allow 2s CI slack."""
    live_agent.decide({"board": ".........", "to_move": "X"}, CELLS)
    start = time.perf_counter()
    live_agent.decide({"board": "...X.....", "to_move": "O"}, CELLS)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 2000
