"""State formatter tests: golden strings for every game plus the generic formatter."""

from __future__ import annotations

import pytest

from verqen_game_ai.games import (
    chess,
    game_2048,
    pacman,
    snake,
    tictactoe,
    tetris,
    tower_defense,
)
from verqen_game_ai.state_formatter import generic_format, get_formatter, state_text


def test_state_text_prefix():
    assert state_text("board: 1") == "<STATE> board: 1"


# --- tictactoe -------------------------------------------------------------

def test_tictactoe_from_string():
    out = tictactoe.format_state("XXO.O....")
    assert out == "<STATE> tictactoe board: XXO.O.... to_move: -"


def test_tictactoe_from_dict_with_side():
    out = tictactoe.format_state({"board": "XXO.O....", "to_move": "o"})
    assert out == "<STATE> tictactoe board: XXO.O.... to_move: O"


def test_tictactoe_from_nested_list():
    board = [["X", "O", None], [".", "X", "."], ["O", ".", "."]]
    out = tictactoe.format_state(board)
    assert out == "<STATE> tictactoe board: XO..X.O.. to_move: -"


def test_tictactoe_rejects_bad_boards():
    with pytest.raises(ValueError, match="9 cells"):
        tictactoe.format_state("XXO.O...")
    with pytest.raises(ValueError, match="invalid cells"):
        tictactoe.format_state("XYZ.O....")
    with pytest.raises(ValueError, match="'board' key"):
        tictactoe.format_state({"to_move": "X"})


# --- snake -----------------------------------------------------------------

def test_snake_from_dict():
    out = snake.format_state(
        {
            "grid": ["...", "#.."],
            "head": [0, 0],
            "food": [1, 2],
            "direction": "up",
            "length": 4,
        }
    )
    assert out == "<STATE> snake grid: .../#.. head: r0c0 food: r1c2 direction: up length: 4"


def test_snake_from_bare_grid():
    out = snake.format_state(["...", "#.."])
    assert out == "<STATE> snake grid: .../#.. head: - food: - direction: - length: -"


def test_snake_requires_grid():
    with pytest.raises(ValueError, match="grid"):
        snake.format_state({"head": [0, 0]})


# --- tetris ----------------------------------------------------------------

def test_tetris_from_dict():
    out = tetris.format_state(
        {"grid": ["..#.", "...."], "piece": "T", "next": "I", "score": 120}
    )
    assert out == "<STATE> tetris grid: ..#./.... piece: T next: I score: 120"


def test_tetris_from_bare_grid():
    out = tetris.format_state(["..#.", "...."])
    assert out == "<STATE> tetris grid: ..#./.... piece: - next: - score: 0"


# --- chess -----------------------------------------------------------------

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_chess_from_fen_string():
    assert chess.format_state(FEN) == f"<STATE> chess fen: {FEN}"


def test_chess_from_dict():
    assert chess.format_state({"fen": FEN}) == f"<STATE> chess fen: {FEN}"


def test_chess_rejects_non_fen():
    with pytest.raises(ValueError, match="FEN"):
        chess.format_state(8)
    with pytest.raises(ValueError, match="FEN"):
        chess.format_state({"board": "rnbqkbnr"})


# --- pacman ----------------------------------------------------------------

def test_pacman_from_dict():
    out = pacman.format_state(
        {
            "maze": ["#####", "#P..#"],
            "pacman": [1, 1],
            "ghosts": [[0, 1], [1, 3]],
            "pellets": 3,
        }
    )
    assert out == "<STATE> pacman maze: #####/#P..# pacman: r1c1 ghosts: 2 pellets: 3"


def test_pacman_ghosts_accept_count_and_bare_maze():
    assert pacman.format_state({"maze": ["###"], "ghosts": 2}) == (
        "<STATE> pacman maze: ### pacman: - ghosts: 2 pellets: 0"
    )
    assert pacman.format_state(["###"]) == (
        "<STATE> pacman maze: ### pacman: - ghosts: 0 pellets: 0"
    )


# --- 2048 ------------------------------------------------------------------

BOARD = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
ROWS_TEXT = "2 4 8 16/0 0 0 0/0 0 0 0/0 0 0 0"


def test_2048_nested_board():
    assert game_2048.format_state(BOARD) == f"<STATE> 2048 board: {ROWS_TEXT}"


def test_2048_flat_board_chunks_into_rows():
    flat = [cell for row in BOARD for cell in row]
    assert game_2048.format_state(flat) == f"<STATE> 2048 board: {ROWS_TEXT}"


def test_2048_dict_score_appends():
    out = game_2048.format_state({"grid": BOARD, "score": 96})
    assert out == f"<STATE> 2048 board: {ROWS_TEXT} score: 96"


def test_2048_bad_shape_raises():
    with pytest.raises(ValueError, match="16 values"):
        game_2048.format_state([1, 2, 3])


# --- tower defense ---------------------------------------------------------

def test_tower_defense_from_dict():
    out = tower_defense.format_state(
        {"wave": 3, "gold": 120, "lives": 15, "enemies": 4, "towers": 6}
    )
    assert out == (
        "<STATE> tower_defense wave: 3 gold: 120 lives: 15 enemies: 4 towers: 6"
    )


def test_tower_defense_lists_and_defaults():
    out = tower_defense.format_state({"enemies": ["a", "b"], "towers": ["t1"]})
    assert out == (
        "<STATE> tower_defense wave: 0 gold: 0 lives: 0 enemies: 2 towers: 1"
    )
    with pytest.raises(ValueError, match="dict"):
        tower_defense.format_state([1, 2, 3])


# --- generic formatter and registry ---------------------------------------

def test_generic_format_golden_strings():
    assert generic_format("hello") == "<STATE> hello"
    assert generic_format({"hp": 10, "name": "hero"}) == "<STATE> hp=10 name=hero"
    assert generic_format([1, 2, 3]) == "<STATE> 1 2 3"
    assert generic_format([["a", "b"], ["c", "d"]]) == "<STATE> a b/c d"
    assert generic_format({"cells": [1, 2], "n": 3}) == "<STATE> cells=1 2 n=3"


def test_get_formatter_returns_game_formatter():
    assert get_formatter("tictactoe") is tictactoe.format_state
    assert get_formatter("TicTacToe") is tictactoe.format_state


def test_get_formatter_unknown_game():
    with pytest.raises(ValueError, match="Available games"):
        get_formatter("battleship")
