# verqen-game-ai benchmarks

Plays a bundled tic-tac-toe puzzle suite with a live VerQen checkpoint and
reports the **optimal-action rate** (how often the agent's move matches the
known-optimal move) plus latency percentiles.

## Dataset

`tictactoe_dataset.jsonl` contains 30 rows:

```json
{"id": "ttt01", "board": "XX..O...O", "to_move": "X", "expected_action": "top_right"}
```

Every position was verified with a full minimax solver before being added:
`expected_action` is always a member of the position's optimal set (immediate
wins, forced blocks and draw-preserving opening moves). The loader re-checks
board legality on every run (9 cells of `X`/`O`/`.`, piece counts consistent
with `to_move`).

## Run

```bash
python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
python benchmarks/run_benchmark.py --limit 10 --fail-under 0.9
```

Options:

| Flag | Meaning |
|------|---------|
| `--checkpoint` | VerQen checkpoint path (defaults to `VERQEN_CHECKPOINT`) |
| `--dataset` | Alternative JSONL dataset |
| `--difficulty` | Agent difficulty 0.0 to 1.0 (default 1.0 = optimal play) |
| `--limit` | Only run the first N rows |
| `--fail-under` | Exit 3 when the optimal-action rate is below this value |

## Output

Prints `optimal`, `confidence` and `latency p50/p95`, and writes a full JSON
report (per-position records included) to `benchmarks/results/benchmark-<stamp>.json`.

The bundled suite is a regression benchmark. The official VerQen numbers in
the repository README (99.1% Tic-Tac-Toe, 93.3% Tetris) come from the full
`model_card.json:suite.environments` evaluation, described in the
[VerQen repository](https://github.com/ZorQelis-AI/verqen).
