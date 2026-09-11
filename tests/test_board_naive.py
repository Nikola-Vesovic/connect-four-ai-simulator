"""Tests that reach into the naive board's internals, so they cannot be shared.

Everything the two implementations have in common lives in test_board_contract.py.
"""

from __future__ import annotations

import random
from itertools import combinations

from core.board_naive import NaiveBoard
from core.rules import COLS, CONNECT, DIRECTIONS, ROWS
from core.types import EMPTY, GameResult, Player


def test_incremental_and_full_scan_detectors_agree_over_random_games():
    """The incremental check used by play() against an independent full-board scan."""
    rng = random.Random(20260910)
    for _ in range(300):
        board = NaiveBoard()
        while not board.is_terminal():
            board.play(rng.choice(board.legal_moves()))
            assert board.winner() == board.scan_for_winner()
        assert board.result() is not GameResult.IN_PROGRESS


def test_win_detection_is_correct_for_every_four_disc_configuration():
    """Exhaustive: all C(42, 4) = 111,930 placements of four discs of one colour, each
    probed from all four discs in turn as if that one had just been played. The detector
    must report a win for exactly the 69 real lines and for nothing else.

    Reaches into the grid directly because gravity cannot produce every configuration.
    """
    lines = {
        frozenset(
            (row + step * d_row, col + step * d_col) for step in range(CONNECT)
        )
        for row in range(ROWS)
        for col in range(COLS)
        for d_row, d_col in DIRECTIONS
        if 0 <= row + (CONNECT - 1) * d_row < ROWS
        and 0 <= col + (CONNECT - 1) * d_col < COLS
    }
    assert len(lines) == 69

    board = NaiveBoard()
    grid = board._grid
    cells = [(row, col) for row in range(ROWS) for col in range(COLS)]

    for combo in combinations(cells, CONNECT):
        for row, col in combo:
            grid[row][col] = Player.ONE
        detections = {board._wins_through(row, col, Player.ONE) for row, col in combo}
        for row, col in combo:
            grid[row][col] = EMPTY

        expected = frozenset(combo) in lines
        assert detections == {expected}, f"{sorted(combo)}: got {detections}, want {expected}"
