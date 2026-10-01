"""Benchmark harness tests: dataset integrity and metric math (no GPU needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from verqen_game_ai.games.tictactoe import DEFAULT_ACTIONS

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_benchmark", _BENCH)
run_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmark)


def test_dataset_loads_and_validates():
    rows = run_benchmark.load_dataset()
    assert len(rows) == 30
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate dataset ids"


def test_dataset_expected_actions_are_legal_cells():
    for row in run_benchmark.load_dataset():
        assert row["expected_action"] in DEFAULT_ACTIONS
        assert row["to_move"] in ("X", "O")
        board = row["board"]
        x_count, o_count = board.count("X"), board.count("O")
        if row["to_move"] == "X":
            assert x_count == o_count
        else:
            assert x_count == o_count + 1


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run_benchmark.percentile(values, 50) == 3.0
    assert run_benchmark.percentile(values, 95) == 5.0
    assert run_benchmark.percentile([], 50) == 0.0


def test_summarize_perfect_predictions():
    records = [
        {"id": "a", "expected": "center", "predicted": "center", "correct": True,
         "confidence": 0.9, "is_optimal": True, "latency_ms": 10.0},
        {"id": "b", "expected": "top_left", "predicted": "top_left", "correct": True,
         "confidence": 0.8, "is_optimal": False, "latency_ms": 30.0},
    ]
    summary = run_benchmark.summarize(records)
    assert summary["optimal_action_rate"] == 1.0
    assert summary["mean_confidence"] == pytest.approx(0.85, abs=1e-4)
    assert summary["is_optimal_rate"] == 0.5
    # nearest-rank: rank = ceil(0.5 * 2) = 1 -> first value.
    assert summary["latency_ms"]["p50"] == 10.0
    assert summary["errors"] == []


def test_summarize_counts_misses():
    records = [
        {"id": "a", "expected": "center", "predicted": "top_left", "correct": False,
         "confidence": 0.5, "is_optimal": False, "latency_ms": 5.0},
        {"id": "b", "expected": "center", "predicted": "center", "correct": True,
         "confidence": 0.9, "is_optimal": True, "latency_ms": 5.0},
    ]
    summary = run_benchmark.summarize(records)
    assert summary["optimal_action_rate"] == 0.5
    assert summary["errors"] == ["a"]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        run_benchmark.summarize([])


def test_main_fails_cleanly_without_checkpoint(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"id": "x1", "board": ".........", "to_move": "X", '
        '"expected_action": "center"}\n',
        encoding="utf-8",
    )
    assert run_benchmark.main(["--dataset", str(dataset)]) == 1
    assert "VERQEN_CHECKPOINT" in capsys.readouterr().err
