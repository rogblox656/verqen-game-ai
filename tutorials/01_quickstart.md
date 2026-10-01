# Tutorial 01: Quickstart

Install the package (no hard dependencies) plus the VerQen engine for live play:

```bash
pip install verqen-game-ai
pip install git+https://github.com/ZorQelis-AI/verqen.git
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
```

On Windows PowerShell use `$env:VERQEN_CHECKPOINT = "C:\path\to\verqen-checkpoint"`,
or copy `.env.example` to `.env` and edit it.

## Your first agent

```python
from verqen_game_ai import VerQenAgent

agent = VerQenAgent.from_pretrained(
    game="tictactoe",
    checkpoint="runs/verqen-t2-remote/release_format",   # optional: reads VERQEN_CHECKPOINT
)
```

`from_pretrained(game=..., checkpoint=..., difficulty=..., device=...)` builds a
`VerQenAgent` for one registered game (run `verqen_game_ai.games.available_games()`
to list them).

## Make a decision

```python
CELLS = ["top_left", "top_center", "top_right",
         "mid_left", "center", "mid_right",
         "bot_left", "bot_center", "bot_right"]

decision = agent.decide(
    board_state="XX.O.....",       # any board format the game's formatter accepts
    available_actions=CELLS,       # legal moves for this position
)

print(decision.action)             # e.g. "top_right"
print(decision.confidence)         # calibrated probability of that move
print(decision.conformal_actions)  # certified set: moves VerQen cannot rule out
print(decision.alternatives)       # top-5 (action, probability) pairs, sorted
print(decision.is_optimal)         # True when confidence > 0.90
print(decision.latency_ms)         # one decide call, milliseconds
```

`decide` makes exactly one engine call: the board goes through the game's
formatter into `"<STATE> ..."` text, the question and legal actions go to VerQen,
and the result comes back as a `GameDecision`.

For tic-tac-toe you can also pass a dict, which carries whose turn it is:

```python
agent.decide({"board": "XX.O.....", "to_move": "O"}, CELLS)
```

## Ask for advice

```python
agent.check("Does X have two in a row?", "XX......O")   # True or False
```

`check` uses VerQen's yes/no (noul) mode with a calibrated `value` field, so you
get a plain `bool` for hint systems and coach prompts.

## Set the difficulty

```python
weak_agent = VerQenAgent.from_pretrained(game="tictactoe", difficulty=0.5)
weak_agent.rng.seed(42)   # optional: reproducible draws
```

`0.0` plays uniformly at random, `0.5` samples from the narrowed conformal
prediction set, `1.0` always takes the engine's argmax. See the
[Difficulty scaling](../README.md#difficulty-scaling) section of the README.

## Try it without a GPU

The test suite runs against an injected fake engine, and
[examples/tictactoe_demo.py](../examples/tictactoe_demo.py) falls back to a tiny
built-in rule engine when `VERQEN_CHECKPOINT` is unset:

```bash
python examples/tictactoe_demo.py
```

Next: [register your own game](02_custom_game.md).
