# Tutorial 03: Train your own game AI

This tutorial covers the data and scheduling side of training with the toolkit
in `verqen_game_ai.training`. The optimizer itself (RLCD: Reinforcement Learning
from Calibrated Decisions) lives in the
[VerQen repository](https://github.com/ZorQelis-AI/verqen); this package builds
the batches and schedules the difficulty so you can get from "my game" to
"trained agent" in about 30 minutes.

## 1. Wrap your environment

```python
from verqen_game_ai.training import GymEnvWrapper

# Option A: wrap an env you already constructed
wrapper = GymEnvWrapper(my_env, game="snake_oa")

# Option B: build it by id (imports gymnasium lazily, pip install "verqen-game-ai[gym]")
wrapper = GymEnvWrapper.from_id("MyEnv-v0", game="snake_oa")

state_text, info = wrapper.reset()          # "<STATE> ..." text, ready for VerQen
state_text, reward, done, truncated, info = wrapper.step(0)
print(wrapper.available_actions())          # labels from the Discrete space
```

Observations become `"<STATE> ..."` through your registered game formatter; no
gymnasium import happens unless you call `from_id`.

## 2. Collect trajectories into prompts

Every training record needs `state`, `question` and `action`; `reward` is
optional:

```python
from verqen_game_ai.training import batch_to_prompts

records = [
    {
        "state": "<STATE> snake_oa grid: .# / H. food: r0c0",
        "question": "Which direction should the snake move next?",
        "action": "up",
        "reward": 1.0,
    },
]
prompts = batch_to_prompts(records)
# ['<STATE> snake_oa grid: .# / H. food: r0c0\nQuestion: ...\nAction: up\nReward: 1.0']
```

`batch_to_prompts` is a pure function: validate it with your own fixtures before
any GPU time is spent.

## 3. Schedule difficulty with a curriculum

```python
from verqen_game_ai import VerQenAgent
from verqen_game_ai.training import Curriculum

curriculum = Curriculum()               # warmup -> beginner -> intermediate -> expert
agent = VerQenAgent(game="snake_oa", difficulty=curriculum.difficulty, ...)

# after each evaluation window:
stage = curriculum.next_stage({"optimal_rate": 0.62})
agent.difficulty = stage.difficulty     # raise the bar when the gate is cleared
```

Each stage has a `min_optimal_rate` gate; `next_stage` advances (and clamps at
the last stage) while recent performance clears it. Stages are plain dataclasses,
so you can pass your own schedule:

```python
from verqen_game_ai.training import Curriculum, Stage

curriculum = Curriculum(stages=(Stage("easy", 0.2, 0.5), Stage("hard", 1.0, 0.9)))
```

## 4. Hand the batch to the VerQen trainer

```python
from verqen_game_ai.training import RLCdTrainer

trainer = RLCdTrainer(game="snake_oa")
batch = trainer.build_batch(records)
trainer.train(batch, iterations=100)
```

`train` deliberately raises a `RuntimeError` that points at the VerQen repository
training docs: this repo ships the batch-building and scheduling toolkit, while
the RLCD optimizer runs with the `verqen` training stack installed
(`pip install git+https://github.com/ZorQelis-AI/verqen.git`).

## 5. Evaluate and publish

- Run your game's own suite, or add positions to a JSONL dataset and use the
  pattern in `benchmarks/run_benchmark.py` (optimal-action rate + latency
  percentiles + `--fail-under` for CI).
- Keep the trained weights out of git (`.gitignore` already excludes
  `*.safetensors`, `*.pt` and `checkpoints/`); publish them via the VerQen repo
  or a release asset, and point `from_pretrained` at them.

Back to the [README](../README.md).
