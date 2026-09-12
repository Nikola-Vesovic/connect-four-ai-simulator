"""Phase 5: the evaluation function.

There is no objectively correct answer for what a non-terminal position is worth, so
these tests do not check that the heuristic is *good*. They check the things that must
hold whatever the weights are, and that break a search when they do not:

  * it never reaches the terminal band, so a search can never prefer a guess to a win,
  * the same position is scored identically on both board representations,
  * a position is worth exactly the opposite to each player,
  * the components are genuinely separable, so their contribution can be measured.

Its actual playing quality is measured, not asserted: the score-versus-depth curve in
`data/results/` is that measurement.
"""

from __future__ import annotations

import random

import pytest

from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from core.rules import CELLS, COLS, CONNECT, ROWS
from core.types import Player, Score
from helpers import random_position
from search.evaluation import (
    CENTER,
    DEFAULT,
    MAX_MAGNITUDE,
    PARITY,
    THREE,
    TWO,
    WINDOWS,
    ZERO,
    Evaluation,
)

THREATS_ONLY = Evaluation(threats=True, center=False, parity=False)
CENTER_ONLY = Evaluation(threats=False, center=True, parity=False)
PARITY_ONLY = Evaluation(threats=False, center=False, parity=True)


def non_terminal_positions(seed: int, count: int) -> list[NaiveBoard]:
    rng = random.Random(seed)
    found = []
    while len(found) < count:
        board = random_position(rng, max_plies=40)
        if not board.is_terminal():
            found.append(board)
    return found


# --------------------------------------------------------------------------- geometry


def test_there_are_sixty_nine_four_in_a_row_windows():
    # Counted independently from the geometry: 7 columns x 3 vertical starts,
    # 6 rows x 4 horizontal starts, and 3 x 4 starts for each of the two diagonals.
    expected = COLS * (ROWS - CONNECT + 1) + ROWS * (COLS - CONNECT + 1) + 2 * 3 * 4
    assert expected == 69
    assert len(WINDOWS) == expected
    assert len(set(WINDOWS)) == expected
    assert all(mask.bit_count() == CONNECT for mask in WINDOWS)


# ------------------------------------------------------------------ the hard constraint


def test_the_heuristic_can_never_reach_the_terminal_band():
    """BUILD_PLAN 4: a heuristic that overlaps the win band lets a search prefer a guess
    to a real win. The bound is computed from the weights, not hoped for."""
    assert MAX_MAGNITUDE < Score.MIN_WIN
    assert MAX_MAGNITUDE == len(WINDOWS) * (THREE + PARITY) + (CELLS // 2) * (COLS // 2) * CENTER


def test_no_random_position_gets_anywhere_near_the_bound():
    worst = 0
    for board in non_terminal_positions(seed=7, count=1500):
        for perspective in Player:
            value = DEFAULT(board, perspective)
            assert abs(value) <= MAX_MAGNITUDE
            worst = max(worst, abs(value))
    assert worst < Score.MIN_WIN


# ------------------------------------------------------------------------- consistency


def test_both_board_implementations_are_scored_identically():
    for board in non_terminal_positions(seed=11, count=800):
        mirror = Bitboard.from_moves(board.history)
        assert DEFAULT(board, Player.ONE) == DEFAULT(mirror, Player.ONE)


def test_a_position_is_worth_the_opposite_to_each_player():
    for board in non_terminal_positions(seed=13, count=800):
        assert DEFAULT(board, Player.ONE) == -DEFAULT(board, Player.TWO)


def test_an_empty_board_is_worth_nothing_to_anyone():
    for evaluation in (DEFAULT, THREATS_ONLY, CENTER_ONLY, PARITY_ONLY, ZERO):
        assert evaluation(NaiveBoard(), Player.ONE) == 0


# ------------------------------------------------------------------------ separability


def test_the_components_add_up():
    """Switchability is not a claim about the flags, it is a claim about the score: the
    full evaluation is exactly the sum of its parts, so each part's contribution to a
    result can be attributed to it."""
    for board in non_terminal_positions(seed=17, count=500):
        parts = sum(e(board, Player.ONE) for e in (THREATS_ONLY, CENTER_ONLY, PARITY_ONLY))
        assert DEFAULT(board, Player.ONE) == parts


def test_switching_everything_off_scores_nothing_anywhere():
    for board in non_terminal_positions(seed=19, count=300):
        assert ZERO(board, Player.ONE) == 0


def test_every_component_actually_changes_some_score():
    boards = non_terminal_positions(seed=23, count=300)
    for component in (THREATS_ONLY, CENTER_ONLY, PARITY_ONLY):
        assert any(component(board, Player.ONE) != 0 for board in boards)


@pytest.mark.parametrize(
    "evaluation,expected",
    [
        (DEFAULT, "full"),
        (ZERO, "zero"),
        (THREATS_ONLY, "threats"),
        (Evaluation(center=False), "threats+parity"),
    ],
)
def test_an_evaluation_names_itself_for_the_results_table(evaluation, expected):
    assert evaluation.name == expected


# ---------------------------------------------------------------------- the components


def test_a_window_holding_both_colours_is_worth_nothing_to_either_side():
    # X X X . on the bottom row: one live three-window.
    live = NaiveBoard.from_moves([0, 6, 1, 6, 2, 5])
    assert THREATS_ONLY(live, Player.ONE) > 0

    # The same position with the gap plugged by O. The window is now dead, and with it
    # the parity bonus that went with it.
    plugged = NaiveBoard.from_moves([0, 6, 1, 6, 2, 3])
    assert THREATS_ONLY(plugged, Player.ONE) < THREATS_ONLY(live, Player.ONE)
    assert PARITY_ONLY(plugged, Player.ONE) == 0


def test_centre_control_is_worth_more_than_the_edge():
    centre = NaiveBoard.from_moves([3, 0])  # X in the centre column, O on the edge
    edge = NaiveBoard.from_moves([0, 3])  # the other way round
    assert CENTER_ONLY(centre, Player.ONE) == CENTER * (COLS // 2)
    assert CENTER_ONLY(edge, Player.ONE) == -CENTER * (COLS // 2)


def test_threat_parity_favours_the_first_player_on_odd_rows():
    """Rows are 0-based from the bottom here, so the 1st row is row 0. The rule of thumb
    is that the first player wants threats on odd rows (1, 3, 5) and the second on even
    rows (2, 4, 6), which in these coordinates is rows 0, 2, 4 against rows 1, 3, 5.

    Three positions, each with exactly one live three-window and nothing else to score:
    """
    # X X X . on row 0 -- odd row, and X is the first player: a bonus.
    ones_threat_on_row_zero = NaiveBoard.from_moves([0, 6, 1, 6, 2, 5])
    assert PARITY_ONLY(ones_threat_on_row_zero, Player.ONE) == PARITY

    # O O O . on row 0 -- same row, but it belongs to the second player: no bonus.
    twos_threat_on_row_zero = NaiveBoard.from_moves([6, 0, 6, 1, 5, 2])
    assert PARITY_ONLY(twos_threat_on_row_zero, Player.ONE) == 0

    # O O O . on row 1 -- even row, and O is the second player: a bonus, for O.
    twos_threat_on_row_one = NaiveBoard.from_moves([0, 1, 2, 0, 6, 1, 6, 2])
    assert PARITY_ONLY(twos_threat_on_row_one, Player.ONE) == -PARITY
