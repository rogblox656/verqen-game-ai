"""Training toolkit: environment adapters, curriculum and the RLCD skeleton.

Everything here is import-safe without gymnasium or verqen: heavy
dependencies are imported inside the functions that need them, and the
pure helpers (state conversion, curriculum math, prompt batching) are fully
testable on their own.
"""

from .curriculum import DEFAULT_STAGES, Curriculum, Stage
from .env_wrapper import GymEnvWrapper, action_names, observation_to_state
from .rlcd_trainer import RLCdTrainer, batch_to_prompts

__all__ = [
    "Curriculum",
    "DEFAULT_STAGES",
    "GymEnvWrapper",
    "RLCdTrainer",
    "Stage",
    "action_names",
    "batch_to_prompts",
    "observation_to_state",
]
