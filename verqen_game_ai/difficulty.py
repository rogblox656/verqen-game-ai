"""Difficulty scaling through conformal abstention.

Turns VerQen's probability distribution over actions into a chosen action
at a human-selected skill level. The mapping (``p`` = engine probabilities,
``d`` = difficulty, ``rng`` = a ``random.Random`` instance):

* ``d >= 1.0``: optimal play. Returns ``argmax p`` (first occurrence wins
  ties, so the result is deterministic).
* ``d <= 0.0``: random play. Uniform over every action, ignoring both the
  probabilities and the prediction set.
* ``0.0 < d < 1.0``: the candidate set starts from the conformal
  prediction set (when given, otherwise all actions) and is narrowed to
  actions whose probability reaches the threshold ``d * max(p)``. The
  final action is sampled from the candidates with weight proportional to
  ``p``. If the threshold leaves no candidate, the best action of the
  starting set is kept.

Higher difficulty can only shrink or keep the candidate set, so play gets
strictly stronger as ``d`` rises. Seeding *rng* makes every draw
reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def clamp_difficulty(difficulty: float) -> float:
    """Clamp *difficulty* into the ``0.0 .. 1.0`` range."""
    return min(1.0, max(0.0, float(difficulty)))


def sample_action(
    probabilities: Mapping[str, float],
    difficulty: float,
    rng: random.Random | None = None,
    prediction_set: Sequence[str] | None = None,
) -> str:
    """Draw one action from *probabilities* at the given *difficulty*.

    Args:
        probabilities: Engine probabilities per action (need not sum to 1).
        difficulty: 0.0 = random, 0.5 = medium, 1.0 = optimal (see module doc).
        rng: Seeded ``random.Random`` for reproducible draws; unseeded when None.
        prediction_set: Certified conformal action set from the engine,
            used to narrow candidates at intermediate difficulties.

    Raises ``ValueError`` when *probabilities* is empty.
    """
    if not probabilities:
        raise ValueError("probabilities is empty; cannot sample an action")
    rng = rng if rng is not None else random.Random()
    d = clamp_difficulty(difficulty)
    keys = list(probabilities)

    if d <= 0.0:
        return rng.choice(keys)
    if d >= 1.0:
        return max(keys, key=lambda a: probabilities[a])

    start = keys
    if prediction_set:
        certified = set(prediction_set)
        narrowed = [a for a in keys if a in certified]
        if narrowed:
            start = narrowed

    max_p = max(probabilities[a] for a in start)
    candidates = [a for a in start if probabilities[a] >= d * max_p]
    if not candidates:
        candidates = [max(start, key=lambda a: probabilities[a])]

    weights = [probabilities[a] for a in candidates]
    if sum(weights) <= 0:
        return rng.choice(candidates)
    return rng.choices(candidates, weights=weights, k=1)[0]
