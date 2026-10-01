# verqen-game-ai

**Plug-and-play neural AI opponents for Python games: 99.1% optimal on Tic-Tac-Toe, 93.3% on Tetris.**

verqen-game-ai turns the [VerQen](https://github.com/ZorQelis-AI/verqen) decision model into a drop-in opponent for Python games. One `decide()` call per move returns the action to play, a calibrated confidence, the certified conformal set and the latency. No prompt engineering, no free text, no LLM bills: a 232M-parameter decision model that reads your board and picks a move from the legal actions you pass in.

```
Board state (dict / list / str)
        |
        v
[VerQenAgent]  formatter -> "<STATE> ..." text
        |   - What is the best action in this position?   [choice over legal actions]
        v
GameDecision(action, confidence, conformal_actions, alternatives, latency_ms, is_optimal)
```

## Why verqen-game-ai

- **Game AI is one of the most-starred categories on GitHub.** Every developer building a Python game wants a smart, trainable opponent, and today that means hand-writing minimax, scripting behaviors or paying for an LLM call per move.
- **VerQen is the only non-autoregressive model with native RL game-environment training.** It was trained on game environments directly (Tic-Tac-Toe 99.1%, Tetris 93.3%, Browser Agent 100% optimal), not fine-tuned from a chat model.
- **Sub-4ms per decision means 250 game decisions per second.** Fast enough to call every frame or inside a real-time game loop with no perceptible delay.
- **Every move ships with an honest confidence and a conformal set**, so you can scale difficulty (below), flag uncertain positions for the player, or route them to a stronger search.

## Benchmarks

Optimal action rate and return of VerQen on game environments:

| Game Environment | Optimal Action Rate | Return |
| :--- | ---: | ---: |
| **Browser Agent** | **100.0%** | +1.773 |
| **Tic-Tac-Toe** | **99.1%** | +0.667 |
| **Grid Navigation** | **96.0%** | +1.563 |
| **Tetris** | **93.3%** | +6.033 |
| **Graph Routing** | **88.0%** | +1.068 |

*Source: VerQen model card (`suite.environments`), measured on RTX 5060 Ti/5090 hardware.*

Reproduce locally with the bundled minimax-verified suite: see [benchmarks/README.md](benchmarks/README.md).

## Install

```bash
pip install verqen-game-ai

# optional extras
pip install "verqen-game-ai[overlay]"   # pygame decision overlay
pip install "verqen-game-ai[gym]"       # gymnasium env wrapper
pip install "verqen-game-ai[gpu]"       # torch runtime for the engine
```

The base package has **no hard dependencies**: formatters, the registry, difficulty scaling and the tests are pure Python. For live play you also need the VerQen engine (it is not on PyPI) and a checkpoint:

```bash
pip install git+https://github.com/ZorQelis-AI/verqen.git
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint    # or copy .env.example to .env
```

## Quickstart

```python
from verqen_game_ai import VerQenAgent

agent = VerQenAgent.from_pretrained(
    game="tictactoe",
    checkpoint="runs/verqen-t2-remote/release_format",   # or omit to read VERQEN_CHECKPOINT
)

decision = agent.decide(
    board_state="XX.O.....",
    available_actions=["top_left", "top_center", "top_right",
                       "mid_left", "center", "mid_right",
                       "bot_left", "bot_center", "bot_right"],
)

print(decision.action)              # top_right
print(decision.confidence)          # 0.97 (calibrated)
print(decision.conformal_actions)   # ["top_right"] (certified set)
print(decision.alternatives)        # [("top_right", 0.97), ("mid_left", 0.02), ...]
print(decision.is_optimal)          # True (confidence > 0.90)
print(decision.latency_ms)          # e.g. 68 (eager)
```

Or ask for advice (yes/no, noul mode):

```python
if agent.check("Does X have two in a row?", "XX......O"):
    print("X is about to win")
```

A runnable version lives in [examples/tictactoe_demo.py](examples/tictactoe_demo.py).

## Supported games

| Game | `game=` | Board input (formatter) | Default actions |
| :--- | :--- | :--- | :--- |
| Tetris | `tetris` | dict `{grid, piece, next, score}` or grid list | `move_left` `move_right` `rotate_left` `rotate_right` `drop` `hold` |
| Snake | `snake` | dict `{grid, head, food, direction, length}` or grid list | `up` `down` `left` `right` |
| Tic-Tac-Toe | `tictactoe` | 9-char string, 3x3 list, or dict `{board, to_move}` | `top_left` ... `bot_right` (9 cells) |
| Chess | `chess` | FEN string or dict `{fen}` | `e2e4` `d2d4` `g1f3` `b1c3` `f1c4` (fallback; pass legal moves from your engine, e.g. python-chess `legal_moves`) |
| Pac-Man | `pacman` | dict `{maze, pacman, ghosts, pellets}` or maze list | `up` `down` `left` `right` `stay` |
| 2048 | `2048` | 4x4 list, flat list of 16, or dict `{grid, score}` | `up` `down` `left` `right` |
| Tower Defense | `tower_defense` | dict `{wave, gold, lives, enemies, towers}` | `build_archer` `build_cannon` `build_slow` `upgrade` `save_gold` |

Every formatter is pure Python and emits canonical `"<STATE> ..."` text. Your own game registers in two lines, see [tutorials/02_custom_game.md](tutorials/02_custom_game.md).

## Difficulty scaling

Set `difficulty` anywhere from `0.0` (random) to `1.0` (optimal) to tune opponent strength. The scaling works through **conformal abstention** on the model's prediction set:

| Difficulty | Behavior |
| :--- | :--- |
| `0.0` | Random play: uniform over all legal actions. |
| `0.5` | Medium: candidates restricted to the conformal prediction set, then to actions whose probability clears `difficulty * max(p)`, then sampled by probability. |
| `1.0` | Optimal play: the engine's argmax (its 99.1% Tic-Tac-Toe move). |

```python
agent = VerQenAgent.from_pretrained(game="tictactoe", difficulty=0.5)
agent.rng.seed(42)   # seeded draws make a difficulty level reproducible
```

The mapping is implemented in `verqen_game_ai/difficulty.py` and documented in its module docstring. Higher difficulty can only shrink the candidate set, so play gets strictly stronger as difficulty rises.

## Advisory questions

`check(question, board_state)` asks a calibrated yes/no question about the position using VerQen's noul mode and returns a plain `bool` (from the engine's `value` field):

```python
agent.check("Is this position a draw?", board)
agent.check("Should I use the hold piece here?", tetris_state)
```

Useful for hint systems, tutorial prompts and "AI coach" features.

## Decision overlay (optional)

With pygame installed, draw VerQen's probability distribution over the moves in real time:

```python
from verqen_game_ai.overlay import create_overlay

surface.blit(create_overlay(decision), (10, 10))   # one bar per action
```

`draw_probability_bars(surface, decision, x, y, width)` draws onto an existing surface. The module imports safely without pygame and raises a helpful error if you call it uninstalled.

## Training your own game

The training toolkit covers the three pieces of a custom game AI:

- **Curriculum**: `Curriculum` schedules difficulty stages (warmup to expert) and advances when your agent's recent optimal rate clears each gate.
- **Env adapter**: `GymEnvWrapper` converts gymnasium-style observations into `"<STATE>"` text.
- **RLCD toolkit**: `batch_to_prompts` builds the state/question/action batches for Reinforcement Learning from Calibrated Decisions (the training stage described in the VerQen repository), and `RLCdTrainer` points you at the VerQen training docs when you call `train()`.

With those, a custom game agent trains in about 30 minutes: see [tutorials/03_train_your_own.md](tutorials/03_train_your_own.md).

## Tutorials

| Tutorial | What it covers |
| :--- | :--- |
| [01 Quickstart](tutorials/01_quickstart.md) | Install, first agent, decide, check, difficulty |
| [02 Custom game](tutorials/02_custom_game.md) | Register your own game in minutes |
| [03 Train your own](tutorials/03_train_your_own.md) | Curriculum, env wrapper, RLCD batch flow |

## Benchmarks (this repo)

```bash
python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
```

Plays the bundled 30-position tic-tac-toe suite (every expected move verified against full minimax) and reports optimal-action rate plus latency p50/p95 to `benchmarks/results/`. See [benchmarks/README.md](benchmarks/README.md).

## Testing

```bash
python -m pytest -q          # fast suite, no GPU/checkpoint needed
python -m pytest -q -m live  # live suite (needs VERQEN_CHECKPOINT)
```

The suite runs entirely on an injected `FakeEngine`; live tests skip when `VERQEN_CHECKPOINT` is unset.

## Project layout

```
verqen_game_ai/
  agent.py            # VerQenAgent + GameDecision
  state_formatter.py  # <STATE> text helpers + formatter lookup
  difficulty.py       # conformal-abstention difficulty scaling
  overlay.py          # optional pygame probability overlay
  games/              # registry + 7 game formatters (pure Python)
  training/           # env wrapper, curriculum, RLCD toolkit
tests/                # FakeEngine suite, no GPU required
benchmarks/           # minimax-verified tic-tac-toe suite + runner
tutorials/            # quickstart, custom game, training
examples/             # runnable tictactoe demo
pretrained/           # checkpoint notes (weights ship with VerQen)
```

## License

Apache-2.0. Copyright ZorQelis AI. VerQen model: [github.com/ZorQelis-AI/verqen](https://github.com/ZorQelis-AI/verqen).
