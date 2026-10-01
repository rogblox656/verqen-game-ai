"""Shared test fixtures for verqen-game-ai.

``FakeEngine`` mimics the real ``VerqenInferenceEngine.decide`` output shape
(``selected``, ``probabilities``, ``prediction_set``, and ``value`` for noul
questions) so the whole suite runs without a GPU, torch or a checkpoint.
"""

from __future__ import annotations

import pytest

from verqen_game_ai import VerQenAgent


class FakeEngine:
    """Deterministic stand-in for the VerQen engine.

    Args:
        selected: Chosen action for choice questions (falls back to the
            highest-probability option, then to the first option).
        probabilities: Explicit ``{action: prob}`` for choice answers;
            defaults to 0.95 on *selected*, the rest sharing 0.05.
        prediction_set: Override for ``prediction_set`` (default: the
            selected action alone).
        value: ``value`` / side for noul (yes/no) questions.
    """

    def __init__(
        self,
        selected: str | None = None,
        probabilities: dict[str, float] | None = None,
        prediction_set: list[str] | None = None,
        value: bool = True,
    ):
        self.selected = selected
        self.probabilities = probabilities
        self.prediction_set = prediction_set
        self.value = value
        self.calls: list[dict] = []

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        options = [str(o) for o in (options or [])]
        self.calls.append(
            {
                "state": state,
                "question": question,
                "options": options,
                "question_type": question_type,
            }
        )
        if question_type == "noul":
            chosen = "yes" if self.value else "no"
            probs = {"yes": 0.9 if self.value else 0.1, "no": 0.1 if self.value else 0.9}
            return {
                "selected": chosen,
                "probabilities": probs,
                "p_correct": 0.9,
                "confidence": 0.9,
                "margin": 0.8,
                "should_escalate": False,
                "prediction_set": [chosen],
                "value": self.value,
                "p_true": probs["yes"],
            }

        probs = dict(self.probabilities) if self.probabilities else None
        if probs is None:
            if self.selected is not None and self.selected in options:
                chosen = self.selected
            else:
                chosen = options[0] if options else ""
            rest = [o for o in options if o != chosen]
            share = 0.05 / max(1, len(rest))
            probs = {o: (0.95 if o == chosen else share) for o in options}
        else:
            if self.selected is not None and self.selected in probs:
                chosen = self.selected
            else:
                chosen = max(probs, key=probs.get)
            if options and chosen not in options:
                in_options = {o: p for o, p in probs.items() if o in options}
                if in_options:
                    probs = in_options
                    chosen = max(probs, key=probs.get)
        prediction_set = list(self.prediction_set) if self.prediction_set else [chosen]
        confidence = float(probs.get(chosen, 0.0))
        return {
            "selected": chosen,
            "probabilities": probs,
            "p_correct": confidence,
            "confidence": confidence,
            "margin": 0.5,
            "should_escalate": False,
            "prediction_set": prediction_set,
        }


@pytest.fixture
def fake_engine_cls():
    return FakeEngine


@pytest.fixture
def make_engine():
    """Factory: ``make_engine(selected="center", prediction_set=[...])``."""

    def _make(**kwargs) -> FakeEngine:
        return FakeEngine(**kwargs)

    return _make


@pytest.fixture
def make_agent():
    """Factory: ``make_agent(difficulty=0.5, engine=FakeEngine(...))``."""

    def _make(**kwargs) -> VerQenAgent:
        kwargs.setdefault("game", "tictactoe")
        kwargs.setdefault("engine", FakeEngine())
        return VerQenAgent(**kwargs)

    return _make
