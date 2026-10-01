"""VerQenAgent: drop-in neural game AI powered by the VerQen decision model.

One ``decide`` call per move, one ``noul`` call per advisory question. The
engine is loaded lazily (or injected), so everything here is testable
without a GPU or the ``verqen`` package.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .difficulty import clamp_difficulty, sample_action
from .engine import load_engine
from .games import get_game

#: Confidence above which a decision is flagged as an optimal-tier move.
#: Heuristic: VerQen's calibrated confidence in the chosen action; above 0.90
#: the model is certain enough that its pick matches expert play in the
#: benchmark suite. It is a proxy, not a guarantee.
OPTIMAL_CONFIDENCE = 0.90


@dataclass
class GameDecision:
    """The outcome of a single ``VerQenAgent.decide`` call.

    Attributes:
        action: The chosen action (difficulty-scaled below difficulty 1.0).
        confidence: Calibrated probability of ``action``.
        conformal_actions: Certified prediction set from the engine (the
            actions VerQen cannot rule out at the target coverage).
        alternatives: Up to 5 ``(action, probability)`` pairs sorted by
            descending probability.
        latency_ms: Wall-clock time of the decide call in milliseconds.
        is_optimal: ``confidence > 0.90`` (see :data:`OPTIMAL_CONFIDENCE`).
    """

    action: str
    confidence: float
    conformal_actions: list[str]
    alternatives: list[tuple[str, float]]
    latency_ms: float
    is_optimal: bool


class VerQenAgent:
    """A neural game AI for one registered game.

    Args:
        game: Registered game name (see ``available_games()``).
        checkpoint: Path to a local VerQen checkpoint. Defaults to the
            ``VERQEN_CHECKPOINT`` environment variable; ignored when
            *engine* is injected.
        difficulty: 0.0 = random, 0.5 = medium, 1.0 = optimal. Values are
            clamped into ``[0.0, 1.0]``.
        device: Device string passed to the VerQen engine (``"auto"``).
        engine: Injected engine with a ``decide`` method (used by tests).
    """

    def __init__(
        self,
        game: str,
        checkpoint: str | None = None,
        difficulty: float = 1.0,
        device: str = "auto",
        engine=None,
    ):
        spec = get_game(game)
        self.game = spec.name
        self.formatter = spec.formatter
        self.question = spec.question
        self.default_actions = list(spec.default_actions)
        self.difficulty = clamp_difficulty(difficulty)
        self.rng = random.Random()
        self.engine = (
            engine if engine is not None else load_engine(checkpoint, device=device, fast=True)
        )

    @classmethod
    def from_pretrained(cls, game: str, checkpoint: str | None = None, **kwargs) -> "VerQenAgent":
        """Build an agent from a pretrained checkpoint (or ``VERQEN_CHECKPOINT``)."""
        return cls(game=game, checkpoint=checkpoint, **kwargs)

    def decide(self, board_state, available_actions) -> GameDecision:
        """Make one game decision in a single engine call.

        Args:
            board_state: Native board representation for the game's formatter.
            available_actions: Legal actions for this position. Falls back
                to the game's default actions when empty.

        Returns:
            A :class:`GameDecision` with the (possibly difficulty-scaled)
            action, its confidence, the conformal set and latency.
        """
        actions = [str(a) for a in (available_actions or self.default_actions)]
        if not actions:
            raise ValueError(f"no available actions for game {self.game!r}")

        start = time.perf_counter()
        state_text = self.formatter(board_state)
        result = self.engine.decide(state_text, self.question, actions)
        elapsed_ms = (time.perf_counter() - start) * 1000

        probabilities = {str(k): float(v) for k, v in (result.get("probabilities") or {}).items()}
        selected = str(result.get("selected", ""))
        conformal = [str(a) for a in (result.get("prediction_set") or [selected])]

        action = selected
        confidence = probabilities.get(selected, float(result.get("confidence") or 0.0))
        if self.difficulty < 1.0 and probabilities:
            action = sample_action(
                probabilities, self.difficulty, self.rng, prediction_set=conformal
            )
            confidence = probabilities.get(action, 0.0)

        alternatives = sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

        return GameDecision(
            action=action,
            confidence=confidence,
            conformal_actions=conformal,
            alternatives=alternatives,
            latency_ms=elapsed_ms,
            is_optimal=confidence > OPTIMAL_CONFIDENCE,
        )

    def check(self, question: str, board_state) -> bool:
        """Ask an advisory yes/no question about the game state (noul mode).

        Uses the engine's calibrated ``value`` field (falling back to the
        selected side for engines that do not return it).
        """
        state_text = self.formatter(board_state)
        result = self.engine.decide(state_text, question, ["yes", "no"], question_type="noul")
        if "value" in result:
            return bool(result["value"])
        return str(result.get("selected", "")).lower() == "yes"
