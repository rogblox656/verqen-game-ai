"""Difficulty sampling tests: seeded determinism and distribution sanity."""

from __future__ import annotations

import random

import pytest

from verqen_game_ai.difficulty import clamp_difficulty, sample_action

PROBS = {"a": 0.55, "b": 0.25, "c": 0.15, "d": 0.05}


def test_empty_probabilities_raise():
    with pytest.raises(ValueError, match="empty"):
        sample_action({}, 1.0, random.Random(0))


def test_clamp_difficulty():
    assert clamp_difficulty(-0.5) == 0.0
    assert clamp_difficulty(0.42) == 0.42
    assert clamp_difficulty(1.7) == 1.0


def test_difficulty_one_is_argmax_always():
    rng = random.Random(0)
    for _ in range(200):
        assert sample_action(PROBS, 1.0, rng) == "a"


def test_difficulty_one_ties_break_by_first_occurrence():
    rng = random.Random(0)
    tied = {"z": 0.5, "m": 0.5}
    assert sample_action(tied, 1.0, rng) == "z"


def test_difficulty_zero_is_near_uniform_over_1000_draws():
    rng = random.Random(1234)
    counts = {action: 0 for action in PROBS}
    for _ in range(1000):
        counts[sample_action(PROBS, 0.0, rng)] += 1
    assert set(counts) == set(PROBS)
    for action, count in counts.items():
        assert 175 <= count <= 325, f"{action} drawn {count}/1000"


def test_difficulty_zero_ignores_prediction_set():
    rng = random.Random(5)
    drawn = {sample_action(PROBS, 0.0, rng, prediction_set=["a"]) for _ in range(300)}
    assert drawn == set(PROBS)


def test_same_seed_same_sequence():
    runs = [
        [sample_action(PROBS, 0.5, random.Random(99)) for _ in range(1)]
        for _ in range(2)
    ]
    # fresh rngs with equal seeds produce equal draws
    assert runs[0] == runs[1]
    a = random.Random(99)
    b = random.Random(99)
    seq_a = [sample_action(PROBS, 0.5, a) for _ in range(50)]
    seq_b = [sample_action(PROBS, 0.5, b) for _ in range(50)]
    assert seq_a == seq_b


def test_intermediate_difficulty_narrows_and_favors_argmax():
    rng = random.Random(7)
    counts = {"a": 0, "b": 0, "c": 0, "d": 0}
    for _ in range(500):
        counts[sample_action(PROBS, 0.5, rng)] += 1
    # threshold 0.5 * 0.55 = 0.275 keeps a and b only
    assert counts["c"] == 0
    assert counts["d"] == 0
    assert counts["a"] > counts["b"]  # weighted by probability within candidates


def test_prediction_set_narrows_candidates():
    rng = random.Random(11)
    # globally worst action wins every draw because it is the certified one.
    for _ in range(50):
        assert sample_action(PROBS, 0.5, rng, prediction_set=["d"]) == "d"


def test_difficulty_one_takes_argmax_even_with_prediction_set():
    rng = random.Random(3)
    for _ in range(50):
        picked = sample_action(PROBS, 1.0, rng, prediction_set=["d"])
        assert picked == "a"  # d >= 1.0 is argmax over all probabilities


def test_all_zero_weights_fall_back_to_a_candidate():
    rng = random.Random(2)
    zeros = {"x": 0.0, "y": 0.0}
    drawn = {sample_action(zeros, 0.5, rng) for _ in range(50)}
    assert drawn <= {"x", "y"}
    assert drawn  # never empty


def test_single_action_any_difficulty():
    rng = random.Random(1)
    for difficulty in (0.0, 0.5, 1.0):
        assert sample_action({"only": 1.0}, difficulty, rng) == "only"
