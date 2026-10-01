# Pretrained checkpoints

This repository ships **no model weights**. The pretrained VerQen policies live
with the VerQen project itself:

- Code and training: [github.com/ZorQelis-AI/verqen](https://github.com/ZorQelis-AI/verqen)
- Weights: [huggingface.co/zorqelis-ai/VerQen](https://huggingface.co/zorqelis-ai/VerQen)

## Point an agent at a local checkpoint

Download or build a checkpoint directory (a `release_format` directory from a
VerQen run works), then either pass it explicitly:

```python
from verqen_game_ai import VerQenAgent

agent = VerQenAgent.from_pretrained(
    game="tictactoe",
    checkpoint="C:/path/to/verqen-checkpoint",
)
```

or set the environment variable once (see `.env.example` at the repository
root):

```bash
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
```

```python
agent = VerQenAgent.from_pretrained(game="tictactoe")   # reads VERQEN_CHECKPOINT
```

The `verqen` package itself is not on PyPI; install it with
`pip install git+https://github.com/ZorQelis-AI/verqen.git`. It is only imported
when you construct a live agent, so everything else in this package works
without it.

## Why no binaries in this repo

Git is the wrong place for hundreds of megabytes of weights. Checkpoints change
independently of the code, are hosted on Hugging Face or your own disk, and are
already excluded by `.gitignore` (`*.safetensors`, `*.pt`, `checkpoints/`).
