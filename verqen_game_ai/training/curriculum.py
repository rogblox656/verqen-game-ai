"""Progressive difficulty curriculum: raise agent strength as play improves.

Pure dataclasses and threshold math, no engine or env involved. Feed
:meth:`Curriculum.next_stage` a stats mapping with an ``optimal_rate``
key (fraction of optimal moves in a recent window) after each training
block; the curriculum advances while performance clears the current
stage's threshold and clamps at the final stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: its difficulty and the gate to leave it."""

    name: str
    difficulty: float
    min_optimal_rate: float


#: Default four-stage schedule: random play to tuned-to-optimal play.
DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage(name="warmup", difficulty=0.0, min_optimal_rate=0.30),
    Stage(name="beginner", difficulty=0.35, min_optimal_rate=0.60),
    Stage(name="intermediate", difficulty=0.70, min_optimal_rate=0.85),
    Stage(name="expert", difficulty=1.0, min_optimal_rate=0.95),
)


@dataclass
class Curriculum:
    """A stage schedule plus the position within it.

    Args:
        stages: Threshold schedule; defaults to :data:`DEFAULT_STAGES`.
        index: Stage to start at (0 by default).
    """

    stages: tuple[Stage, ...] = DEFAULT_STAGES
    index: int = 0

    def __post_init__(self):
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        if not 0 <= self.index < len(self.stages):
            raise ValueError(f"index {self.index} out of range for {len(self.stages)} stages")

    @property
    def current(self) -> Stage:
        """The active stage."""
        return self.stages[self.index]

    @property
    def difficulty(self) -> float:
        """Difficulty of the active stage, ready for ``VerQenAgent(difficulty=...)``."""
        return self.current.difficulty

    def next_stage(self, stats: Mapping[str, float]) -> Stage:
        """Advance while performance clears the current gate; return the active stage.

        ``stats["optimal_rate"]`` (missing = 0.0) is compared against the
        current stage's ``min_optimal_rate``; each cleared gate moves the
        curriculum forward one stage, never past the last.
        """
        rate = float(stats.get("optimal_rate", 0.0))
        while (
            self.index < len(self.stages) - 1
            and rate >= self.stages[self.index].min_optimal_rate
        ):
            self.index += 1
        return self.current

    def reset(self) -> Stage:
        """Return to the first stage."""
        self.index = 0
        return self.current
