"""Gymnasium-style environment adapter: observations to VerQen ``<STATE>`` text.

Pure helpers (:func:`observation_to_state`, :func:`action_names`) work on any
object without gymnasium installed; :class:`GymEnvWrapper` wraps an already
constructed env and only imports gymnasium in :meth:`GymEnvWrapper.from_id`.
"""

from __future__ import annotations

from ..state_formatter import generic_format, get_formatter


def observation_to_state(game: str, observation) -> str:
    """Convert an env *observation* to ``<STATE>`` text for *game*.

    Uses the registered game formatter; unknown game names raise
    ``ValueError`` from the registry.
    """
    return get_formatter(game)(observation)


def action_names(space, names=None) -> list[str]:
    """Action labels for a Discrete-like *space*.

    *names* wins when given (and must match ``space.n``); otherwise labels
    are ``"0" .. "n-1"`` from the space's ``n`` attribute. Raises
    ``ValueError`` for spaces without ``n``.
    """
    if names is not None:
        labels = [str(name) for name in names]
        n = getattr(space, "n", None)
        if n is not None and len(labels) != int(n):
            raise ValueError(f"got {len(labels)} action names for a space with n={n}")
        return labels
    n = getattr(space, "n", None)
    if n is None:
        raise ValueError(f"cannot derive action names from {type(space).__name__}: no 'n' attribute")
    return [str(i) for i in range(int(n))]


class GymEnvWrapper:
    """Wrap a gymnasium-style env so VerQen sees ``<STATE>`` text.

    The wrapper is duck-typed: any object with ``reset()`` and ``step()``
    works. ``reset`` must return ``(observation, info)`` and ``step`` must
    return ``(observation, reward, terminated, truncated, info)``.
    """

    def __init__(self, env, game: str, names=None):
        self.env = env
        self.game = game
        self._names = list(names) if names is not None else None

    @classmethod
    def from_id(cls, env_id: str, game: str, **make_kwargs) -> "GymEnvWrapper":
        """Create the env with ``gymnasium.make`` (imports gymnasium lazily)."""
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required for GymEnvWrapper.from_id but is not installed. "
                'Install it with: pip install "verqen-game-ai[gym]"'
            ) from exc
        return cls(gym.make(env_id, **make_kwargs), game)

    def reset(self, **kwargs):
        """Reset the env; returns ``(state_text, info)``."""
        observation, info = self.env.reset(**kwargs)
        return observation_to_state(self.game, observation), info

    def step(self, action):
        """Step the env; returns ``(state_text, reward, terminated, truncated, info)``."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation_to_state(self.game, observation), reward, terminated, truncated, info

    def available_actions(self) -> list[str]:
        """Action labels for the wrapped env's action space (or given *names*)."""
        return action_names(self.env.action_space, self._names)
