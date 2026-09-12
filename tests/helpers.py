"""Position generators and independent oracles shared by the test modules.

The oracle for "is there a win here" is `NaiveBoard.scan_for_winner`, the full-board scan
from Phase 1 -- never the incremental detector an agent itself sees. A test that asks the
code under test whether the code under test is right proves nothing.
"""

from __future__ import annotations

import random

from core.board_naive import NaiveBoard
from core.types import Move


def random_position(rng: random.Random, max_plies: int = 30) -> NaiveBoard:
    board = NaiveBoard()
    for _ in range(rng.randrange(max_plies)):
        if board.is_terminal():
            break
        board.play(rng.choice(board.legal_moves()))
    return board


def winning_columns(board: NaiveBoard) -> list[Move]:
    """Columns that win at once for the player to move, per the independent scan."""
    me = board.player_to_move
    found = []
    for col in board.legal_moves():
        board.play(col)
        if board.scan_for_winner() is me:
            found.append(col)
        board.undo()
    return found


def safe_columns(board: NaiveBoard) -> list[Move]:
    """Columns after which the opponent cannot win with their very next disc."""
    safe = []
    for col in board.legal_moves():
        board.play(col)
        if board.is_terminal() or not winning_columns(board):
            safe.append(col)
        board.undo()
    return safe
