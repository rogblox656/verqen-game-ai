"""Game registry tests: completeness, spec shape and registration rules."""

from __future__ import annotations

import pytest

from verqen_game_ai.games import (
    GameSpec,
    available_games,
    chess,
    game_2048,
    get_game,
    pacman,
    register_game,
    snake,
    tictactoe,
    tetris,
    tower_defense,
)

EXPECTED_GAMES = ["2048", "chess", "pacman", "snake", "tetris", "tictactoe", "tower_defense"]
MODULES = {
    "tetris": tetris,
    "snake": snake,
    "tictactoe": tictactoe,
    "chess": chess,
    "pacman": pacman,
    "2048": game_2048,
    "tower_defense": tower_defense,
}


def test_registry_lists_every_bundled_game():
    assert available_games() == EXPECTED_GAMES


@pytest.mark.parametrize("name", EXPECTED_GAMES)
def test_spec_shape(name):
    spec = get_game(name)
    assert isinstance(spec, GameSpec)
    assert spec.name == name
    assert callable(spec.formatter)
    assert isinstance(spec.default_actions, tuple)
    assert len(spec.default_actions) > 0
    assert len(set(spec.default_actions)) == len(spec.default_actions), "duplicate actions"
    assert all(isinstance(a, str) and a for a in spec.default_actions)
    assert spec.question.strip().endswith("?")


@pytest.mark.parametrize("name", EXPECTED_GAMES)
def test_spec_wired_to_module_constants(name):
    module = MODULES[name]
    spec = get_game(name)
    assert spec.formatter is module.format_state
    assert spec.default_actions == module.DEFAULT_ACTIONS
    assert spec.question == module.QUESTION


@pytest.mark.parametrize("name", EXPECTED_GAMES)
def test_default_formatter_outputs_state_prefix(name):
    spec = get_game(name)
    sample = {
        "tictactoe": ".........",
        "chess": "8/8/8/8/8/8/8/8 w - - 0 1",
        "snake": [[".", "."]],
        "tetris": [[".", "."]],
        "pacman": [["#", "#"]],
        "2048": [[2, 0, 0, 0]] * 4,
        "tower_defense": {"wave": 1},
    }[name]
    assert spec.formatter(sample).startswith("<STATE> ")


def test_get_game_case_insensitive():
    assert get_game("TicTacToe").name == "tictactoe"


def test_register_custom_game_and_duplicate():
    register_game("mygame", generic_formatter, ["a", "b"], "Pick one?")
    try:
        assert "mygame" in available_games()
        with pytest.raises(ValueError, match="already registered"):
            register_game("mygame", generic_formatter, ["a"], "Pick?")
        register_game("mygame", generic_formatter, ["a"], "Pick?", replace=True)
        assert get_game("mygame").default_actions == ("a",)
    finally:
        from verqen_game_ai.games import _REGISTRY

        _REGISTRY.pop("mygame", None)


def test_register_validation():
    with pytest.raises(ValueError, match="non-empty"):
        register_game("  ", generic_formatter, ["a"], "Pick?")
    with pytest.raises(ValueError, match="empty"):
        register_game("noactions", generic_formatter, [], "Pick?")


def generic_formatter(board_state):
    return f"<STATE> mygame {board_state}"
