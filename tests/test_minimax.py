"""Phase 5: the score scale, and depth-limited minimax.

Minimax is the first search in this project, and from here on it is also the *reference*:
alpha-beta, move ordering, transposition tables and negascout are all required to return
the value this agent returns, only faster. So these tests check two different kinds of
thing.

The policy, as properties over thousands of random positions rather than a few hand-picked
ones -- Phase 4 showed why: a defect on 1.9% of moves survived a 5,000-game tournament and
died on the first property test. The oracle is `NaiveBoard.scan_for_winner`, which is not
what the agent sees.

And the accounting, because the node counts are a headline result: leaves must match the
game tree exactly, and the count must not depend on the evaluation function or on move
ordering -- a search with no pruning cannot be affected by either, and that is worth
proving here, because in Phase 6 it stops being true and becomes the thing we measure.
"""

from __future__ import annotations

import random

import pytest

from agents.minimax import MinimaxAgent
from agents.greedy_agent import GreedyAgent
from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from core.rules import CELLS
from core.types import Score
from experiments import reference
from helpers import random_position, safe_columns, winning_columns
from search.evaluation import DEFAULT, MAX_MAGNITUDE, ZERO
from search.ordering import CENTER_FIRST
from search.stats import Budget, SearchStats

# Found by searching random positions for the shape, then frozen (see the two tests that
# use them). Move strings are 1-based, the standard benchmark format.
WIN_IN_ONE_OR_THREE = "1454626174511516234"
EVERY_MOVE_LOSES = "363326452"


def column_values(agent: MinimaxAgent, board, depth: int) -> dict[int, int]:
    """What each legal column is worth to the player to move, one ply below the root."""
    values = {}
    for col in board.legal_moves():
        board.play(col)
        values[col] = agent._min_value(board, depth - 1, 1, SearchStats())
        board.undo()
    return values


# ------------------------------------------------------------------------- the scale


def test_the_scale_prefers_a_fast_win_and_a_slow_loss():
    assert Score.win_in(1) > Score.win_in(3) > Score.DRAW
    assert Score.DRAW > Score.loss_in(5) > Score.loss_in(1)
    assert Score.loss_in(0) == Score.LOSS and Score.win_in(0) == Score.WIN


def test_the_terminal_band_is_wide_enough_for_the_longest_game_and_no_wider():
    """42 plies is the whole board, so no win or loss can score closer to zero than
    MIN_WIN -- which is exactly the room the heuristic is allowed to use."""
    assert Score.MAX_MATE_DISTANCE == CELLS
    assert Score.win_in(CELLS) == Score.MIN_WIN
    assert Score.loss_in(CELLS) == Score.MAX_LOSS
    assert MAX_MAGNITUDE < Score.MIN_WIN


# -------------------------------------------------------------------------- the policy


def test_at_depth_one_it_always_takes_an_available_win():
    agent = MinimaxAgent(1)
    rng = random.Random(3)
    tested = 0
    for _ in range(2500):
        board = random_position(rng)
        if board.is_terminal():
            continue
        wins = winning_columns(board)
        if not wins:
            continue
        tested += 1
        move, stats = agent.choose_move(board)
        assert move in wins
        assert stats.value == Score.win_in(1)
    assert tested > 300, f"only {tested} positions with a win in one -- test is too weak"


def test_at_depth_two_it_never_hands_over_an_immediate_win():
    """'Blocks an immediate threat', stated so it is still meaningful when blocking is
    impossible: if a column exists after which the opponent cannot win at once, it has
    to play one."""
    agent = MinimaxAgent(2)
    rng = random.Random(5)
    tested = 0
    for _ in range(2500):
        board = random_position(rng)
        if board.is_terminal():
            continue
        safe = safe_columns(board)
        if not safe or len(safe) == len(board.legal_moves()):
            continue  # nothing to get wrong: everything is safe, or nothing is
        tested += 1
        move, _ = agent.choose_move(board)
        assert move in safe
    assert tested > 300, f"only {tested} positions where safety was a choice"


def test_a_deeper_search_still_takes_the_win_in_one():
    """The mate-distance term as a property. Without it every winning line scores the
    same, and a deep search is free to take one three plies away instead."""
    agent = MinimaxAgent(3)
    rng = random.Random(29)
    tested = 0
    for _ in range(700):
        board = random_position(rng)
        if board.is_terminal():
            continue
        wins = winning_columns(board)
        if not wins:
            continue
        tested += 1
        move, stats = agent.choose_move(board)
        assert move in wins
        assert stats.value == Score.win_in(1)
    assert tested > 80, f"only {tested} positions with a win in one"


def test_it_prefers_the_win_in_one_to_the_more_central_win_in_three():
    """The frozen witness for mate distance, chosen so the tie-break works against the
    right answer: column 6 wins at once, columns 5 and 7 win two plies later, and column
    5 is the one centre order reaches first. Drop the distance term and all three score
    WIN, so the agent plays column 5 and postpones a win it already has."""
    board = NaiveBoard.from_move_string(WIN_IN_ONE_OR_THREE)
    agent = MinimaxAgent(3)

    values = column_values(agent, board, 3)
    assert values[5] == Score.win_in(1)
    assert values[4] == values[6] == Score.win_in(3)
    assert CENTER_FIRST.index(4) < CENTER_FIRST.index(5)  # the trap is armed

    move, stats = agent.choose_move(board)
    assert move == 5
    assert stats.value == Score.win_in(1)


def test_when_every_move_loses_it_loses_as_slowly_as_it_can():
    """The other half of mate distance. Column 1 loses in four plies, every other column
    loses in two, and column 4 is the one the centre tie-break reaches first -- so
    without the distance term the agent resigns two plies earlier than it has to."""
    board = NaiveBoard.from_move_string(EVERY_MOVE_LOSES)
    agent = MinimaxAgent(4)

    values = column_values(agent, board, 4)
    assert values[0] == Score.loss_in(4)
    assert all(value == Score.loss_in(2) for col, value in values.items() if col != 0)
    assert CENTER_FIRST.index(3) < CENTER_FIRST.index(0)  # the trap is armed

    move, stats = agent.choose_move(board)
    assert move == 0
    assert stats.value == Score.loss_in(4)


def test_two_ply_minimax_with_no_evaluation_is_exactly_the_greedy_agent():
    """Phase 4 claimed greedy *is* a two-ply minimax with a trivial evaluation. Here is
    that claim as a test rather than a remark.

    With the evaluation switched off every quiet move scores 0, a move that hands over an
    immediate win scores loss_in(2), a win scores win_in(1), and centre order breaks all
    the ties -- which is greedy's policy, column for column. It is also the control for
    the score-versus-depth curve: whatever depth-2 minimax gains over greedy has to be
    the evaluation function, because with it removed the two agents are the same agent.
    """
    minimax = MinimaxAgent(2, evaluation=ZERO)
    greedy = GreedyAgent()
    rng = random.Random(43)
    compared = 0
    for _ in range(800):
        board = random_position(rng)
        if board.is_terminal():
            continue
        compared += 1
        assert minimax.choose_move(board)[0] == greedy.choose_move(board)[0]
    assert compared > 500  # 560 of the 800 draws are non-terminal, measured


def test_it_is_deterministic():
    rng = random.Random(31)
    for _ in range(60):
        board = random_position(rng)
        if board.is_terminal():
            continue
        first, _ = MinimaxAgent(2, seed=1).choose_move(board)
        second, _ = MinimaxAgent(2, seed=99).choose_move(board)
        assert first == second


def test_both_board_implementations_search_to_the_same_answer():
    rng = random.Random(37)
    agent = MinimaxAgent(3)
    for _ in range(40):
        board = random_position(rng)
        if board.is_terminal():
            continue
        naive_move, naive_stats = agent.choose_move(board)
        fast_move, fast_stats = agent.choose_move(Bitboard.from_moves(board.history))
        assert naive_move == fast_move
        assert naive_stats.value == fast_stats.value
        assert naive_stats.nodes == fast_stats.nodes


# ---------------------------------------------------------------------- the accounting


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_it_visits_the_whole_game_tree_to_its_depth(depth):
    """No pruning means no surprises: from an empty board the first win is impossible
    before ply 7, so every leaf sits at the depth limit. Leaves must be 7^d -- the same
    numbers the Phase 2 perft produced -- and nodes the sum of the levels above."""
    _, stats = MinimaxAgent(depth, evaluation=ZERO).choose_move(Bitboard())
    assert stats.leaves == 7**depth
    assert stats.nodes == sum(7**level for level in range(1, depth + 1))
    assert stats.depth == depth


def test_the_node_count_does_not_depend_on_the_evaluation_function():
    """A search with no cutoffs cannot be steered by what it finds. This is what makes
    the node-count baseline in data/results/ measurable with the cheap evaluation."""
    rng = random.Random(41)
    for _ in range(25):
        board = random_position(rng)
        if board.is_terminal():
            continue
        _, full = MinimaxAgent(4, evaluation=DEFAULT).choose_move(board)
        _, zero = MinimaxAgent(4, evaluation=ZERO).choose_move(board)
        assert full.nodes == zero.nodes
        assert full.leaves == zero.leaves


def test_the_node_count_does_not_depend_on_move_ordering(monkeypatch):
    """True here and false from Phase 6 on: ordering is worth nothing without pruning,
    and everything with it. Same board, reversed column order, identical counts and
    identical value."""
    import agents.minimax as minimax_module

    board = NaiveBoard.from_move_string("4455")
    _, forward = MinimaxAgent(4).choose_move(board)

    monkeypatch.setattr(minimax_module, "CENTER_FIRST", tuple(reversed(minimax_module.CENTER_FIRST)))
    _, backward = MinimaxAgent(4).choose_move(board)

    assert forward.nodes == backward.nodes
    assert forward.leaves == backward.leaves
    assert forward.value == backward.value


def test_a_terminal_line_stops_the_search_short():
    """Once a line is won it is not expanded, so a tree containing wins is smaller than
    the full 7^d. This is the only reason the deep baselines in data/results/ are not
    exactly powers of seven."""
    board = NaiveBoard.from_moves([0, 6, 1, 6, 2, 5])  # X X X . on the bottom row
    assert all(board.can_play(col) for col in range(7))  # no full column to confuse it
    _, stats = MinimaxAgent(3, evaluation=ZERO).choose_move(board)
    assert stats.leaves < 7**3


# -------------------------------------------------------------------------- the budget


def test_the_budget_can_only_make_the_search_shallower():
    agent = MinimaxAgent(4, evaluation=ZERO)
    _, capped = agent.choose_move(Bitboard(), Budget(depth=2))
    assert capped.depth == 2 and capped.nodes == 7 + 49

    _, raised = agent.choose_move(Bitboard(), Budget(depth=9))
    assert raised.depth == 4  # a ceiling, not an instruction


def test_a_search_with_no_depth_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one ply"):
        MinimaxAgent(0)


def test_it_reports_a_value_on_the_score_scale():
    _, stats = MinimaxAgent(2).choose_move(NaiveBoard())
    assert stats.value is not None
    assert abs(stats.value) <= MAX_MAGNITUDE
    assert stats.move == 3


# ----------------------------------------------------------------- the reference values


def test_the_reference_set_was_generated_by_the_code_that_is_here_now():
    """The stored values depend on the score scale and the evaluation weights. If either
    moves, the file is stale and the Phase 6 gate would fail for the wrong reason."""
    data = reference.load()
    assert data["metadata"]["signature"] == reference.signature()
    assert data["metadata"]["records"] == len(data["records"])


def test_the_reference_set_covers_the_whole_game():
    data = reference.load()
    depths = {record["depth"] for record in data["records"]}
    plies = {record["plies"] for record in data["records"]}
    assert depths == {1, 2, 3, 4, 5, 6, 7}
    assert min(plies) == 0 and max(plies) >= 25
    for record in data["records"]:
        assert Score.LOSS <= record["value"] <= Score.WIN
        assert not Bitboard.from_move_string(record["moves"]).is_terminal()


def test_minimax_reproduces_its_own_reference_values():
    """The shallow half of the file, in the suite. The whole file is replayed by
    `python -m experiments.reference --verify`, which is the Phase 6 gate."""
    checked, wrong = reference.verify(reference.load(), max_depth=3)
    assert checked == 600
    assert wrong == []
