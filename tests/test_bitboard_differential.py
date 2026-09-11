"""Phase 2 gate: the bitboard must be indistinguishable from the naive reference.

Every optimisation in this thesis is value-preserving, so correctness reduces to agreement.
Here that means the two representations must answer every question identically, at every
ply of every game.
"""

from __future__ import annotations

import random

import pytest

from core.bitboard import _HEIGHT, Bitboard
from core.board_naive import NaiveBoard
from core.rules import COLS, ROWS
from core.types import Player

CELL_COORDS = [(row, col) for row in range(ROWS) for col in range(COLS)]
SENTINEL_MASK = sum(1 << (col * _HEIGHT + ROWS) for col in range(COLS))

# Leaves of the game tree at each depth, computed by this project and cross-checked between
# the two implementations. Up to depth 6 these are just 7^d. Depth 7 is 7^7 - 7: in the
# seven lines that play one column six times over, that column is full at ply 6, so only
# six moves remain instead of seven.
POSITION_COUNTS = {1: 7, 2: 49, 3: 343, 4: 2_401, 5: 16_807, 6: 117_649, 7: 823_536}


def count_leaves(board, depth: int) -> int:
    if depth == 0 or board.is_terminal():
        return 1
    total = 0
    for col in board.legal_moves():
        board.play(col)
        total += count_leaves(board, depth - 1)
        board.undo()
    return total


def observable_state(board) -> tuple:
    return (
        board.moves_played,
        board.player_to_move,
        tuple(board.legal_moves()),
        board.winner(),
        board.is_full(),
        board.is_terminal(),
        board.result(),
        tuple(board.cell(row, col) for row, col in CELL_COORDS),
    )


# --------------------------------------------------------------------------- lockstep


def test_random_games_agree_at_every_ply():
    """Drive both boards through the same games, comparing all 42 cells after each move."""
    rng = random.Random(20260911)
    plies = 0
    for _ in range(2_000):
        naive, bits = NaiveBoard(), Bitboard()
        assert observable_state(naive) == observable_state(bits)
        while not naive.is_terminal():
            col = rng.choice(naive.legal_moves())
            naive.play(col)
            bits.play(col)
            plies += 1
            assert observable_state(naive) == observable_state(bits), (
                f"diverged after {naive.history}\n{naive}\n\n{bits}"
            )
    assert plies > 40_000  # guards against the loop silently doing nothing


def test_undo_keeps_the_two_representations_in_step():
    rng = random.Random(99)
    for _ in range(200):
        naive, bits = NaiveBoard(), Bitboard()
        while not naive.is_terminal():
            col = rng.choice(naive.legal_moves())
            for board in (naive, bits):
                board.play(col)
                board.undo()
                board.play(col)
            assert observable_state(naive) == observable_state(bits)
        for _ in range(naive.moves_played):
            naive.undo()
            bits.undo()
        assert observable_state(naive) == observable_state(bits) == observable_state(Bitboard())


# --------------------------------------------------------------------------- position counts


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_position_counts_agree_with_the_reference(depth):
    expected = POSITION_COUNTS[depth]
    assert count_leaves(NaiveBoard(), depth) == expected
    assert count_leaves(Bitboard(), depth) == expected


def test_position_count_at_depth_seven():
    """The first depth where a column fills up, so the first real test of move generation.

    Reference walk omitted here only because it is slow; both were checked to agree through
    depth 8 (5,686,266 leaves) outside the test suite.
    """
    assert count_leaves(Bitboard(), 7) == POSITION_COUNTS[7]


# --------------------------------------------------------------------------- sentinel row


def test_sentinel_row_is_never_set():
    rng = random.Random(31337)
    for _ in range(500):
        board = Bitboard()
        while not board.is_terminal():
            board.play(rng.choice(board.legal_moves()))
            assert not board._mask & SENTINEL_MASK


def test_sentinel_row_is_what_prevents_vertical_wraparound():
    """The same four discs, repacked six bits per column, do align. The seventh bit is
    therefore load-bearing and not merely padding."""
    board = Bitboard.from_moves([4, 3, 3, 3, 4, 3, 3, 0, 3, 0])
    assert board.winner() is None

    discs = [
        (row, col)
        for row, col in CELL_COORDS
        if board.cell(row, col) == Player.ONE
    ]
    without_sentinel = sum(1 << (col * ROWS + row) for row, col in discs)
    pairs = without_sentinel & (without_sentinel >> 1)
    assert pairs & (pairs >> 2)
