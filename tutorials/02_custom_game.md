# Tutorial 02: Add your own game

Every game needs three things: a formatter that turns your board into
`"<STATE> ..."` text, a list of default actions, and the question VerQen should
answer. Register them and `VerQenAgent` works unchanged.

## 1. Write a formatter

The formatter is a pure function: native board representation in, canonical text
out. Use `verqen_game_ai.state_formatter.generic_format` if your state is already
a dict/list/str, or build the string yourself for full control.

```python
from verqen_game_ai.state_formatter import state_text

def format_snake_oa(board):
    # board is a dict like {"grid": [[".", "#"], ["H", "."]], "food": [0, 0]}
    grid = "/".join("".join(row) for row in board["grid"])
    food = f"r{board['food'][0]}c{board['food'][1]}"
    return state_text(f"snake_oa grid: {grid} food: {food}")
```

Rules of thumb:

- Always start with `state_text(...)` (or a literal `"<STATE> "` prefix): VerQen
  reads a single text field.
- Be deterministic. Same board in, same text out, every time.
- Keep it compact: `"<STATE> snake_oa grid: .# / H. food: r0c0"` beats a JSON dump.

## 2. Register the game

```python
from verqen_game_ai.games import available_games, register_game

register_game(
    "snake_oa",
    format_snake_oa,
    default_actions=["up", "down", "left", "right"],
    question="Which direction should the snake move next?",
)

assert "snake_oa" in available_games()
```

Names are case-insensitive; registering a duplicate raises `ValueError` unless
you pass `replace=True`.

## 3. Play

```python
from verqen_game_ai import VerQenAgent

agent = VerQenAgent(game="snake_oa", checkpoint=...)   # or engine=... to inject
decision = agent.decide(
    board_state={"grid": [[".", "#"], ["H", "."]], "food": [0, 0]},
    available_actions=["up", "down", "left", "right"],
)
print(decision.action, decision.confidence)
```

Pass `available_actions` per position whenever your legal moves vary; the
registered defaults are only a fallback.

## Checklist

- [ ] Formatter output starts with `<STATE> ` and is stable for the same board.
- [ ] Default actions are unique, non-empty strings.
- [ ] Question ends with `?` and names the decision you want.
- [ ] You can decide and check without an engine by injecting a fake with a
      `decide` method (see `tests/conftest.py` for the expected shape).

Next: [train a policy for your game](03_train_your_own.md).
