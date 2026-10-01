"""Run the bundled tic-tac-toe puzzle suite through verqen-game-ai and record metrics.

Usage::

    python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
    python benchmarks/run_benchmark.py --limit 10 --fail-under 0.9

Loads ``tictactoe_dataset.jsonl`` (hand-verified win, block and positional
positions with known-optimal moves), plays every position with a live
agent, and writes a JSON report to ``benchmarks/results/`` with:

* optimal-action rate (agent move == expected optimal move)
* latency p50 / p95 per decision
* mean confidence and the list of missed positions

The bundled dataset is a regression suite. The official VerQen benchmark
numbers in the README come from the full suite.environments eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "tictactoe_dataset.jsonl"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(HERE.parent))  # allow running without pip install

from verqen_game_ai import VerQenAgent  # noqa: E402
from verqen_game_ai.games.tictactoe import DEFAULT_ACTIONS  # noqa: E402

CELLS = list(DEFAULT_ACTIONS)
REQUIRED_FIELDS = {"id", "board", "to_move", "expected_action"}


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    """Load and validate the JSONL dataset (id, board, to_move, expected_action)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path.name}:{lineno}: missing fields {sorted(missing)}")
            board = row["board"]
            if len(board) != 9 or set(board) - set("XO."):
                raise ValueError(f"{path.name}:{lineno}: bad board {board!r}")
            if row["to_move"] not in ("X", "O"):
                raise ValueError(f"{path.name}:{lineno}: bad to_move {row['to_move']!r}")
            if row["expected_action"] not in CELLS:
                raise ValueError(
                    f"{path.name}:{lineno}: bad expected_action {row['expected_action']!r}"
                )
            x_count, o_count = board.count("X"), board.count("O")
            if row["to_move"] == "X" and x_count != o_count:
                raise ValueError(f"{path.name}:{lineno}: X to move but counts differ")
            if row["to_move"] == "O" and x_count != o_count + 1:
                raise ValueError(f"{path.name}:{lineno}: O to move but counts differ")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: dataset is empty")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate ids")
    return rows


def check_row(agent: VerQenAgent, row: dict) -> dict:
    """Run one dataset position through the agent and time it."""
    start = time.perf_counter()
    decision = agent.decide({"board": row["board"], "to_move": row["to_move"]}, CELLS)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "id": row["id"],
        "expected": row["expected_action"],
        "predicted": decision.action,
        "correct": decision.action == row["expected_action"],
        "confidence": round(decision.confidence, 4),
        "is_optimal": decision.is_optimal,
        "latency_ms": round(elapsed_ms, 3),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-int(p * len(ordered)) // 100)))
    return ordered[rank - 1]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-record results into a metrics report."""
    total = len(records)
    if total == 0:
        raise ValueError("no records to summarize")
    correct = sum(1 for r in records if r["correct"])
    latencies = [r["latency_ms"] for r in records]
    confidences = [r["confidence"] for r in records]
    return {
        "total": total,
        "optimal_action_rate": round(correct / total, 4),
        "mean_confidence": round(sum(confidences) / total, 4),
        "is_optimal_rate": round(sum(1 for r in records if r["is_optimal"]) / total, 4),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "max": round(max(latencies), 3),
        },
        "errors": [r["id"] for r in records if not r["correct"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="VerQen checkpoint path")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="agent difficulty 0.0-1.0 (default 1.0 = optimal play)")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N rows")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="exit 3 when the optimal-action rate is below this value")
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    try:
        agent = VerQenAgent(game="tictactoe", checkpoint=args.checkpoint,
                            difficulty=args.difficulty)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    started = time.time()
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(check_row(agent, row))
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)

    report = {
        "dataset": str(args.dataset.name),
        "game": "tictactoe",
        "difficulty": args.difficulty,
        "rows": len(records),
        "wall_seconds": round(time.time() - started, 2),
        "metrics": summarize(records),
        "records": records,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"benchmark-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print()
    print(f"dataset      {report['dataset']} ({report['rows']} rows)")
    print(f"optimal      {metrics['optimal_action_rate']:.1%}")
    print(f"confidence   {metrics['mean_confidence']:.3f}")
    print(f"latency p50  {metrics['latency_ms']['p50']} ms  p95 {metrics['latency_ms']['p95']} ms")
    print(f"report       {out_path}")
    if metrics["errors"]:
        print(f"misses ({len(metrics['errors'])}): {', '.join(metrics['errors'])}")

    if args.fail_under is not None and metrics["optimal_action_rate"] < args.fail_under:
        print(
            f"FAIL: optimal-action rate {metrics['optimal_action_rate']:.3f} "
            f"< {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
