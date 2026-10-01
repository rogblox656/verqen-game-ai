"""VerQenAgent tests: decide, check, from_pretrained and GameDecision fields.

Every test runs against the injected FakeEngine; no GPU or checkpoint.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from verqen_game_ai import GameDecision, VerQenAgent
from verqen_game_ai.agent import OPTIMAL_CONFIDENCE
from verqen_game_ai.games.tictactoe import DEFAULT_ACTIONS, QUESTION

CELLS = list(DEFAULT_ACTIONS)
ALL_PROBS = {cell: 0.0 for cell in CELLS}


def test_gamedecision_fields():
    names = [f.name for f in dataclasses.fields(GameDecision)]
    assert names == [
        "action",
        "confidence",
        "conformal_actions",
        "alternatives",
        "latency_ms",
        "is_optimal",
    ]


def test_decide_returns_decision_fields(make_agent, make_engine):
    engine = make_engine(selected="center", prediction_set=["center", "top_left"])
    agent = make_agent(engine=engine, difficulty=1.0)
    decision = agent.decide(".........", CELLS)

    assert isinstance(decision, GameDecision)
    assert decision.action == "center"
    assert decision.confidence == pytest.approx(0.95)
    assert decision.conformal_actions == ["center", "top_left"]
    assert decision.alternatives[0] == ("center", pytest.approx(0.95))
    assert decision.latency_ms >= 0
    assert decision.is_optimal is True  # 0.95 > 0.90


def test_decide_is_one_engine_call(make_agent, make_engine):
    engine = make_engine()
    agent = make_agent(engine=engine)
    agent.decide(".........", CELLS)

    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["state"].startswith("<STATE> tictactoe board:")
    assert call["question"] == QUESTION
    assert call["options"] == CELLS
    assert call["question_type"] == "choice"


def test_decide_is_optimal_below_threshold(make_engine):
    probs = dict(ALL_PROBS)
    probs["center"] = OPTIMAL_CONFIDENCE  # boundary is exclusive: not > 0.90
    agent = VerQenAgent(game="tictactoe", engine=make_engine(probabilities=probs))
    decision = agent.decide(".........", CELLS)
    assert decision.is_optimal is False


def test_alternatives_sorted_and_capped(make_agent, make_engine):
    actions = [f"move_{i}" for i in range(7)]
    probs = {a: 0.01 * (i + 1) for i, a in enumerate(actions)}  # ascending input
    agent = make_agent(engine=make_engine(probabilities=probs))
    decision = agent.decide(".........", actions)

    assert len(decision.alternatives) == 5
    values = [p for _, p in decision.alternatives]
    assert values == sorted(values, reverse=True)
    assert decision.alternatives[0][0] == "move_6"


def test_conformal_defaults_to_selected(make_agent, make_engine):
    agent = make_agent(engine=make_engine(selected="top_right"))
    decision = agent.decide(".........", CELLS)
    assert decision.conformal_actions == ["top_right"]


def test_empty_actions_fall_back_to_defaults(make_agent, make_engine):
    engine = make_engine()
    agent = make_agent(engine=engine)
    agent.decide(".........", [])
    assert engine.calls[0]["options"] == CELLS


def test_difficulty_one_takes_engine_argmax(make_agent, make_engine):
    probs = {**ALL_PROBS, "mid_left": 0.60, "center": 0.40}
    agent = make_agent(engine=make_engine(probabilities=probs), difficulty=1.0)
    for _ in range(5):
        assert agent.decide(".........", CELLS).action == "mid_left"


def test_difficulty_zero_is_seeded_uniform(make_agent, make_engine):
    def draws(seed):
        agent = make_agent(engine=make_engine(), difficulty=0.0)
        agent.rng = random.Random(seed)
        return [agent.decide(".........", CELLS).action for _ in range(20)]

    first, second = draws(7), draws(7)
    assert first == second  # seeded rng makes difficulty deterministic
    assert set(first) <= set(CELLS)
    assert len(set(first)) > 1  # difficulty 0.0 does not degenerate to one action


def test_difficulty_clamped(make_agent):
    assert make_agent(difficulty=2.0).difficulty == 1.0
    assert make_agent(difficulty=-1.0).difficulty == 0.0


def test_unknown_game_raises_with_available_list(make_agent):
    with pytest.raises(ValueError, match="Unknown game"):
        make_agent(game="half-life")


def test_check_true(make_agent, make_engine):
    engine = make_engine(value=True)
    agent = make_agent(engine=engine)
    assert agent.check("Does X have two in a row?", "XX......O") is True

    call = engine.calls[0]
    assert call["question_type"] == "noul"
    assert call["options"] == ["yes", "no"]
    assert call["state"].startswith("<STATE> tictactoe")


def test_check_false(make_agent, make_engine):
    agent = make_agent(engine=make_engine(value=False))
    assert agent.check("Is this position a draw?", ".........") is False


def test_check_prefers_value_over_selected():
    class ValueOnlyEngine:
        def decide(self, state, question, options=None, question_type="choice", cost=None):
            return {"selected": "no", "probabilities": {"yes": 0.4, "no": 0.6}, "value": True}

    agent = VerQenAgent(game="tictactoe", engine=ValueOnlyEngine())
    assert agent.check("Safe to sacrifice the rook?", ".........") is True


def test_check_falls_back_to_selected_without_value():
    class SelectedOnlyEngine:
        def decide(self, state, question, options=None, question_type="choice", cost=None):
            return {"selected": "yes", "probabilities": {"yes": 0.8, "no": 0.2}}

    agent = VerQenAgent(game="tictactoe", engine=SelectedOnlyEngine())
    assert agent.check("Is the center free?", ".........") is True


def test_from_pretrained_with_injected_engine(make_engine):
    engine = make_engine(selected="center")
    agent = VerQenAgent.from_pretrained(game="TicTacToe", difficulty=0.5, engine=engine)
    assert agent.game == "tictactoe"
    assert agent.difficulty == 0.5
    assert agent.engine is engine


def test_from_pretrained_requires_checkpoint(monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="VERQEN_CHECKPOINT"):
        VerQenAgent.from_pretrained(game="tictactoe")
