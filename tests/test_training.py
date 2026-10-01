"""Training toolkit tests: curriculum transitions, batching and env adapters.

All pure-Python: no gymnasium, pygame or checkpoint required.
"""

from __future__ import annotations

import importlib.util

import pytest

from verqen_game_ai.training import (
    DEFAULT_STAGES,
    Curriculum,
    GymEnvWrapper,
    RLCdTrainer,
    Stage,
    action_names,
    batch_to_prompts,
    observation_to_state,
)

HAS_GYMNASIUM = importlib.util.find_spec("gymnasium") is not None


# --- curriculum ------------------------------------------------------------

def test_default_curriculum_shape():
    assert len(DEFAULT_STAGES) == 4
    curriculum = Curriculum()
    assert curriculum.current.name == "warmup"
    assert curriculum.difficulty == 0.0


def test_next_stage_stays_below_gate():
    curriculum = Curriculum()
    stage = curriculum.next_stage({"optimal_rate": 0.10})
    assert stage.name == "warmup"
    assert curriculum.index == 0


def test_next_stage_advances_one_stage():
    curriculum = Curriculum()
    stage = curriculum.next_stage({"optimal_rate": 0.35})
    assert stage.name == "beginner"
    assert stage.difficulty == 0.35


def test_next_stage_advances_through_all_gates():
    curriculum = Curriculum()
    stage = curriculum.next_stage({"optimal_rate": 1.0})
    assert stage.name == "expert"
    assert stage.difficulty == 1.0


def test_next_stage_clamps_at_last_stage():
    curriculum = Curriculum()
    curriculum.next_stage({"optimal_rate": 1.0})
    assert curriculum.next_stage({"optimal_rate": 1.0}).name == "expert"


def test_next_stage_missing_stats_hold_position():
    curriculum = Curriculum()
    curriculum.next_stage({"optimal_rate": 0.40})
    stage = curriculum.next_stage({})
    assert stage.name == "beginner"  # missing key reads as 0.0, no advance


def test_reset_returns_to_first_stage():
    curriculum = Curriculum()
    curriculum.next_stage({"optimal_rate": 1.0})
    assert curriculum.reset().name == "warmup"


def test_curriculum_validation():
    with pytest.raises(ValueError, match="at least one stage"):
        Curriculum(stages=())
    with pytest.raises(ValueError, match="out of range"):
        Curriculum(index=5)


def test_custom_stage_schedule():
    stages = (Stage("fast", 0.2, 0.5), Stage("full", 1.0, 0.9))
    curriculum = Curriculum(stages=stages)
    assert curriculum.next_stage({"optimal_rate": 0.6}).name == "full"


# --- batching --------------------------------------------------------------

def test_batch_to_prompts_golden():
    prompts = batch_to_prompts(
        [
            {
                "state": "<STATE> tictactoe board: .........",
                "question": "Best square?",
                "action": "center",
                "reward": 1.0,
            }
        ]
    )
    assert prompts == [
        "<STATE> tictactoe board: .........\n"
        "Question: Best square?\n"
        "Action: center\n"
        "Reward: 1.0"
    ]


def test_batch_to_prompts_without_reward():
    prompts = batch_to_prompts(
        [{"state": "<STATE> x", "question": "q", "action": "a"}]
    )
    assert prompts == ["<STATE> x\nQuestion: q\nAction: a"]


def test_batch_to_prompts_missing_key_names_index():
    with pytest.raises(ValueError, match=r"record 1: missing keys \['action'\]"):
        batch_to_prompts(
            [
                {"state": "<STATE> x", "question": "q", "action": "a"},
                {"state": "<STATE> y", "question": "q"},
            ]
        )


def test_trainer_build_batch_and_train_error():
    trainer = RLCdTrainer(game="tictactoe")
    records = [{"state": "<STATE> x", "question": "q", "action": "a"}]
    assert trainer.build_batch(records) == batch_to_prompts(records)

    with pytest.raises(RuntimeError, match="github.com/ZorQelis-AI/verqen") as exc:
        trainer.train(records, iterations=2)
    assert "RLCD" in str(exc.value)


# --- env adapter -----------------------------------------------------------

def test_observation_to_state_uses_registry():
    out = observation_to_state("tictactoe", "XX......O")
    assert out == "<STATE> tictactoe board: XX......O to_move: -"
    with pytest.raises(ValueError, match="Unknown game"):
        observation_to_state("unknown", "x")


class _Discrete:
    def __init__(self, n):
        self.n = n


class _Box:
    def __init__(self):
        self.shape = (3,)


def test_action_names_from_space():
    assert action_names(_Discrete(3)) == ["0", "1", "2"]


def test_action_names_override():
    assert action_names(_Discrete(2), ["left", "right"]) == ["left", "right"]
    with pytest.raises(ValueError, match="n=2"):
        action_names(_Discrete(2), ["a", "b", "c"])


def test_action_names_without_n_raises():
    with pytest.raises(ValueError, match="no 'n' attribute"):
        action_names(_Box())


class _FakeEnv:
    def __init__(self):
        self.action_space = _Discrete(4)

    def reset(self):
        return ".........", {"seed": 1}

    def step(self, action):
        return "XX.......", 1.0, True, False, {"action": action}


def test_env_wrapper_reset_and_step():
    wrapper = GymEnvWrapper(_FakeEnv(), game="tictactoe")
    state, info = wrapper.reset()
    assert state == "<STATE> tictactoe board: ......... to_move: -"
    assert info == {"seed": 1}

    state, reward, terminated, truncated, info = wrapper.step(0)
    assert state == "<STATE> tictactoe board: XX....... to_move: -"
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert wrapper.available_actions() == ["0", "1", "2", "3"]


@pytest.mark.skipif(HAS_GYMNASIUM, reason="gymnasium installed")
def test_from_id_raises_helpful_error_without_gymnasium():
    with pytest.raises(ImportError, match="gymnasium is required") as exc:
        GymEnvWrapper.from_id("CartPole-v1", game="tictactoe")
    assert "verqen-game-ai[gym]" in str(exc.value)
