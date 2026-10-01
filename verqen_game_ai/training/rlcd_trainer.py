"""RLCD training toolkit skeleton for custom games.

RLCD is VerQen's **Reinforcement Learning from Calibrated Decisions** stage
(see the VerQen repository README): exact expected reward over the option
set, a proper scoring rule that keeps probabilities honest, and
group-relative policy gradients on multi-step environments.

The flow for your own game:

1. Roll out your env (optionally through :class:`GymEnvWrapper`) and collect
   ``(state, question, actions, action, reward)`` trajectories.
2. Build training prompts with :func:`batch_to_prompts` (pure, tested).
3. Run the actual update with the VerQen training stack: this package ships
   the data plumbing only, and :meth:`RLCdTrainer.train` points you at the
   VerQen repo when asked to train.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

_VERQEN_REPO = "https://github.com/ZorQelis-AI/verqen"
_INSTALL_HINT = "pip install git+https://github.com/ZorQelis-AI/verqen.git"


def batch_to_prompts(batch: Iterable[Mapping[str, Any]]) -> list[str]:
    """Turn decision records into VerQen training prompts (pure function).

    Each record needs ``state`` (``<STATE> ...`` text or raw state that a
    registered formatter already converted), ``question`` and ``action``.
    Optional ``reward`` is appended when present. Raises ``ValueError``
    naming the offending index when a required key is missing.
    """
    prompts: list[str] = []
    for index, record in enumerate(batch):
        missing = {"state", "question", "action"} - set(record)
        if missing:
            raise ValueError(f"record {index}: missing keys {sorted(missing)}")
        prompt = f"{record['state']}\nQuestion: {record['question']}\nAction: {record['action']}"
        if "reward" in record:
            prompt += f"\nReward: {record['reward']}"
        prompts.append(prompt)
    return prompts


class RLCdTrainer:
    """Skeleton that builds batches and hands training to the VerQen repo.

    Args:
        game: Registered game name (or a custom registered game).
        checkpoint: Checkpoint the trained policy will start from.
    """

    def __init__(self, game: str, checkpoint: str | None = None):
        self.game = game
        self.checkpoint = checkpoint

    def build_batch(self, trajectories: Iterable[Mapping[str, Any]]) -> list[str]:
        """Convert trajectories to prompt strings via :func:`batch_to_prompts`."""
        return batch_to_prompts(trajectories)

    def train(self, batch: Iterable[Mapping[str, Any]], *, iterations: int = 1) -> None:
        """Run RLCD training. Always raises: this repo ships no training loop.

        The real optimizer lives in the VerQen repository; the error names
        the install command and the training docs, and the batch itself is
        already prepared with :meth:`build_batch`.
        """
        try:
            import verqen  # noqa: F401
        except ImportError:
            hint = (
                "RLCD training needs the verqen training stack, which is not installed. "
                f"Install it with: {_INSTALL_HINT}. "
            )
        else:
            hint = (
                "This package ships batch building only; the RLCD optimizer lives in the "
                "VerQen repository. "
            )
        raise RuntimeError(
            hint
            + f"See the training docs at {_VERQEN_REPO} "
            f"(README section 'RLCD: Reinforcement Learning from Calibrated Decisions'), "
            f"then run your batch of {len(list(batch))} prompts there. "
            f"iterations requested: {iterations}."
        )
