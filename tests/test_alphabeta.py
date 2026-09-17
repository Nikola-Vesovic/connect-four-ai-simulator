"""Phase 6: negamax, and alpha-beta pruning.

The phase has two halves and they are tested differently, on purpose.

The **negamax refactor** is supposed to be a reshaping of Phase 5's code and nothing else,
so it is held to the strictest standard available: with pruning switched off it must
return the same value *and visit the same nodes, leaf for leaf*, as `MinimaxAgent`. Any
difference at all means the refactor changed the search rather than its shape.

**Alpha-beta** is held to a weaker standard, because it is supposed to be different: the
value must survive, the node count must not. Only the value is under contract, which is
why the frozen reference set stores it and not the tree size.

Then there is the part that is new from this phase onwards. A pruned tree depends on the
evaluation function and on the move ordering -- what can be cut depends on what has
already been found and on the order it was found in. Phase 5 proved the *opposite* for
minimax, and those two tests are mirrored here to show the property flipping, because it
is the reason every node count from now on has to be quoted with both.
"""

from __future__ import annotations

import random

import pytest

from agents.alphabeta import _BEYOND_LOSS, _BEYOND_WIN, NegamaxAgent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from core.types import Score
from experiments import reference
from helpers import random_position, safe_columns, winning_columns
from search.deepening import DIRECT, ITERATIVE
from search.evaluation import DEFAULT, ZERO
from search.ordering import DYNAMIC, LEFT, RANDOM, STATIC
from search.scouting import NO_SCOUT, SCOUT
from search.stats import Budget
from search.transposition import NO_TABLE, TABLE
from test_minimax import EVERY_MOVE_LOSES, WIN_IN_ONE_OR_THREE


def positions(seed: int, count: int):
    rng = random.Random(seed)
    made = 0
    while made < count:
        board = random_position(rng)
        if board.is_terminal():
            continue
        made += 1
        yield board


# --------------------------------------------------------------- what makes negamax legal


def test_the_score_scale_is_antisymmetric():
    """Negamax reads a child's value from the child's side and negates it, so a win at
    distance p must be the exact negative of a loss at distance p. If it were not, mate
    distance would drift by a ply every time the sign flipped."""
    for plies in range(Score.MAX_MATE_DISTANCE + 1):
        assert Score.win_in(plies) == -Score.loss_in(plies)


def test_switching_pruning_off_is_a_window_no_value_can_escape():
    """Which is why there is no second code path: if alpha is never raised, the window
    stays outside the scale at every node and `alpha >= beta` can never be true."""
    assert _BEYOND_LOSS < Score.LOSS and _BEYOND_WIN > Score.WIN


# ------------------------------------------------------- the refactor changed nothing


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_negamax_without_pruning_is_the_phase_five_minimax_node_for_node(depth):
    """`NO_TABLE` is not incidental. From Phase 8 the transposition table is on by
    default, and it removes nodes even with pruning off -- that is the whole point of it,
    and it is measured elsewhere. This test is about the negamax *refactor*, so it holds
    every later optimisation at its Phase 6 setting."""
    plain = MinimaxAgent(depth, evaluation=DEFAULT)
    folded = NegamaxAgent(depth, pruning=False, evaluation=DEFAULT, transposition=NO_TABLE)
    for board in positions(seed=101 + depth, count=30):
        before_move, before = plain.choose_move(board)
        after_move, after = folded.choose_move(board)
        assert after.value == before.value
        assert after.nodes == before.nodes
        assert after.leaves == before.leaves
        assert after_move == before_move


def test_without_pruning_nothing_is_ever_cut_off():
    for board in positions(seed=103, count=20):
        _, stats = NegamaxAgent(4, pruning=False).choose_move(board)
        assert stats.cutoffs == 0


# ------------------------------------------------------------ alpha-beta preserves value


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_alpha_beta_returns_the_minimax_value(depth):
    """The Phase 6 gate in miniature. The full version is 1,050 frozen records replayed by
    `python -m experiments.reference --verify --agent alphabeta`."""
    plain = MinimaxAgent(depth, evaluation=DEFAULT)
    pruned = NegamaxAgent(depth, pruning=True, evaluation=DEFAULT)
    for board in positions(seed=107 + depth, count=25 if depth < 5 else 8):
        assert pruned.choose_move(board)[1].value == plain.choose_move(board)[1].value


def test_alpha_beta_chooses_the_same_column_as_minimax_and_not_only_the_same_value():
    """Stronger than the contract requires, and worth pinning because it is what makes
    every Phase 5 playing-strength number carry over unchanged instead of needing to be
    re-measured.

    It is not luck. The root searches with an infinite window, so a move that fails low
    returns a value at most alpha, which is the best found so far -- and the root only
    replaces its best on a strict improvement, exactly as minimax does. A move that could
    have tied can therefore never displace the one already chosen.
    """
    for depth in (2, 3, 4):
        plain = MinimaxAgent(depth, evaluation=DEFAULT)
        pruned = NegamaxAgent(depth, pruning=True, evaluation=DEFAULT)
        for board in positions(seed=113 + depth, count=25):
            assert pruned.choose_move(board)[0] == plain.choose_move(board)[0]


def test_the_value_survives_even_a_deliberately_bad_ordering():
    """Soundness does not depend on the ordering; only speed does. Left-to-right and a
    per-node shuffle are the two orders slide 148 warns about, and the answer has to be
    identical under both anyway."""
    board = NaiveBoard.from_move_string("4455")
    _, good = NegamaxAgent(5, ordering=STATIC, transposition=NO_TABLE).choose_move(board)

    for bad in (LEFT, RANDOM):
        _, worse = NegamaxAgent(
            5, ordering=bad, transposition=NO_TABLE, seed=1
        ).choose_move(board)
        assert worse.value == good.value, bad.name
        assert worse.nodes != good.nodes, bad.name  # it really did search another tree


# ------------------------------------------------------------------------ and cuts nodes


def test_alpha_beta_visits_far_fewer_nodes_and_says_how_many_times_it_cut():
    """`stats.cutoffs` has existed since Phase 4 and has been zero ever since. This is the
    phase in which it finally has something to count."""
    plain_total = pruned_total = cutoffs = 0
    for board in positions(seed=127, count=10):
        _, plain = MinimaxAgent(5, evaluation=DEFAULT).choose_move(board)
        _, pruned = NegamaxAgent(5, pruning=True, evaluation=DEFAULT).choose_move(board)
        assert pruned.nodes <= plain.nodes
        plain_total += plain.nodes
        pruned_total += pruned.nodes
        cutoffs += pruned.cutoffs
    assert cutoffs > 0
    assert pruned_total < plain_total / 5  # measured over the reference set: about 12x


def test_the_empty_board_tree_is_the_size_it_was_measured_to_be():
    """Pinned exactly, with the full evaluation -- and now with the ordering and the
    transposition setting named too, because from Phase 7 the ordering is a setting and
    from Phase 8 so is the table.

    The three rows are kept side by side on purpose, and none of them is ever loosened.
    `STATIC/NO_TABLE` is Phase 6's measurement, unchanged to the node, which is what
    proves that making the ordering and the table into parameters did not disturb the
    search they were measured with. `DYNAMIC/NO_TABLE` is Phase 7's -- including at depth
    5, where it is *worse*, because the empty board is alpha-beta's best case (thesis
    point P11) and a learned ordering has little to add there.

    The third row is the phase's result on this position and it goes the other way: the
    empty board is the most symmetric position on the board, which is exactly where the
    same position is reachable by the most different move orders, so it is where a
    transposition table has the most to find. It wins at every depth from 4.

    The fourth row is Phase 9's, and it is the first row that is *worse than the one above
    it on purpose*. Iterating to depth d searches depths 1..d, and what the shallow
    iterations buy back -- a root move, killer slots, history counts and table entries for
    the very tree about to be searched -- does not cover the re-search at these depths. It
    starts to at depth 11 on this position (`deepening_on_empty_board.csv`), and the
    purpose of the algorithm is not to be cheaper anyway: it is to be stoppable.

    The fifth row is Phase 10's, and it is the one that has to be read a column at a time.
    NegaScout is *behind* at depth 3 and level at 1 and 2 -- a null window can save
    nothing at a node whose children are horizon nodes, and a re-search there is pure
    loss -- and then it takes 9% at depth 5 and 15% at depth 6, growing to 39% at depth 10
    (`scout_on_empty_board.csv`). A single row of six numbers containing both a loss and a
    gain is the reason the depth sweep is reported rather than one headline.
    """
    phase6 = {1: (7, 0), 2: (20, 6), 3: (81, 11), 4: (181, 59), 5: (819, 139), 6: (1526, 491)}
    phase7 = {1: (7, 0), 2: (20, 6), 3: (81, 11), 4: (180, 59), 5: (850, 148), 6: (1514, 468)}
    phase8 = {1: (7, 0), 2: (20, 6), 3: (81, 11), 4: (170, 49), 5: (695, 121), 6: (1127, 277)}
    phase9 = {1: (7, 0), 2: (27, 6), 3: (108, 17), 4: (278, 67), 5: (877, 171), 6: (2004, 456)}
    phase10 = {1: (7, 0), 2: (20, 6), 3: (84, 12), 4: (172, 50), 5: (650, 121), 6: (960, 262)}

    for ordering, table, deepening, scouting, expected in (
        (STATIC, NO_TABLE, DIRECT, NO_SCOUT, phase6),
        (DYNAMIC, NO_TABLE, DIRECT, NO_SCOUT, phase7),
        (DYNAMIC, TABLE, DIRECT, NO_SCOUT, phase8),
        (DYNAMIC, TABLE, ITERATIVE, NO_SCOUT, phase9),
        (DYNAMIC, TABLE, DIRECT, SCOUT, phase10),
    ):
        for depth, (nodes, cutoffs) in expected.items():
            _, stats = NegamaxAgent(
                depth, evaluation=DEFAULT, ordering=ordering, transposition=table,
                deepening=deepening, scouting=scouting,
            ).choose_move(Bitboard())
            assert (stats.nodes, stats.cutoffs) == (nodes, cutoffs), (
                f"{ordering.name}/{table.name}/{deepening.name}/{scouting.name} "
                f"at depth {depth}"
            )


# ---------------------------------------------- the property that flips in this phase


def test_the_pruned_node_count_does_depend_on_the_evaluation_function():
    """The mirror of `test_the_node_count_does_not_depend_on_the_evaluation_function`.
    A search that cuts on what it has found can be steered by what it finds, so a node
    count quoted without naming the evaluation is not a measurement of anything."""
    differed = 0
    for board in positions(seed=131, count=40):
        _, full = NegamaxAgent(5, evaluation=DEFAULT).choose_move(board)
        _, zero = NegamaxAgent(5, evaluation=ZERO).choose_move(board)
        differed += full.nodes != zero.nodes
    assert differed > 20, f"only {differed} of 40 positions differed"

def test_the_pruned_node_count_does_depend_on_move_ordering():
    """The mirror of `test_the_node_count_does_not_depend_on_move_ordering`. Phase 5's
    version of this test asserted equality; here the same experiment has to come out
    unequal, which is the entire justification for Phase 7 existing.

    The direction is measured on this position rather than claimed in general: the naive
    left-to-right order costs 7,281 nodes against centre-first's 4,602 here, 58% more for
    the same answer -- and Phase 7's killers and history bring the same answer down to
    1,923, which is 2.4x below the static centre order rather than above it.
    """
    board = NaiveBoard.from_move_string("4455")
    _, centre = NegamaxAgent(6, ordering=STATIC, transposition=NO_TABLE).choose_move(board)
    _, naive = NegamaxAgent(6, ordering=LEFT, transposition=NO_TABLE).choose_move(board)
    _, learned = NegamaxAgent(6, ordering=DYNAMIC, transposition=NO_TABLE).choose_move(board)

    assert naive.value == centre.value == learned.value
    assert naive.nodes > centre.nodes > learned.nodes


# ------------------------------------------------ the policy, unchanged from Phase 5


def test_at_depth_one_it_still_takes_an_available_win():
    agent = NegamaxAgent(1)
    rng = random.Random(3)
    tested = 0
    for _ in range(1500):
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
    assert tested > 150, f"only {tested} positions with a win in one"  # 192, measured


def test_at_depth_two_it_still_never_hands_over_an_immediate_win():
    agent = NegamaxAgent(2)
    rng = random.Random(5)
    tested = 0
    for _ in range(1500):
        board = random_position(rng)
        if board.is_terminal():
            continue
        safe = safe_columns(board)
        if not safe or len(safe) == len(board.legal_moves()):
            continue
        tested += 1
        assert agent.choose_move(board)[0] in safe
    assert tested > 200, f"only {tested} positions where safety was a choice"


@pytest.mark.parametrize("pruning", [False, True])
def test_mate_distance_survives_the_sign_flips(pruning):
    """The frozen Phase 5 witnesses, replayed. These are the two positions the refactor
    could plausibly have broken: negamax negates a value at every level, and a mate
    distance that drifted by one ply would show up here as the wrong column."""
    quick = NegamaxAgent(3, pruning=pruning).choose_move(
        NaiveBoard.from_move_string(WIN_IN_ONE_OR_THREE)
    )
    assert quick[0] == 5 and quick[1].value == Score.win_in(1)

    slow = NegamaxAgent(4, pruning=pruning).choose_move(
        NaiveBoard.from_move_string(EVERY_MOVE_LOSES)
    )
    assert slow[0] == 0 and slow[1].value == Score.loss_in(4)


# -------------------------------------------------------------------- the usual seams


def test_it_is_deterministic():
    for board in positions(seed=31, count=30):
        assert (
            NegamaxAgent(3, seed=1).choose_move(board)[0]
            == NegamaxAgent(3, seed=99).choose_move(board)[0]
        )


def test_both_board_implementations_search_to_the_same_answer():
    agent = NegamaxAgent(4)
    for board in positions(seed=37, count=25):
        naive_move, naive = agent.choose_move(board)
        fast_move, fast = agent.choose_move(Bitboard.from_moves(board.history))
        assert naive_move == fast_move
        assert naive.value == fast.value
        assert naive.nodes == fast.nodes


def test_the_budget_can_only_make_the_search_shallower():
    agent = NegamaxAgent(6, evaluation=ZERO)
    _, capped = agent.choose_move(Bitboard(), Budget(depth=2))
    assert capped.depth == 2
    _, raised = agent.choose_move(Bitboard(), Budget(depth=9))
    assert raised.depth == 6


def test_a_search_with_no_depth_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one ply"):
        NegamaxAgent(0)


def test_the_name_says_whether_pruning_is_on():
    assert NegamaxAgent(8).name == "alphabeta-d8"
    assert NegamaxAgent(8, pruning=False).name == "negamax-d8"
    assert NegamaxAgent(8, evaluation=ZERO).name == "alphabeta-d8-zero"


# ----------------------------------------------------------------- the reference values


def test_negamax_reproduces_the_reference_values_and_the_reference_node_counts():
    """The refactor gate. Node counts are compared too, which is only meaningful because
    this mode has no cutoffs. The full 1,050 records:
    `python -m experiments.reference --verify --agent negamax`."""
    factory, check_nodes = reference.SEARCHES["negamax"]
    assert check_nodes
    checked, wrong = reference.verify(
        reference.load(), agent_factory=factory, max_depth=3, check_nodes=True
    )
    assert checked == 600
    assert wrong == []


def test_alpha_beta_reproduces_the_reference_values():
    """The pruning gate. Values only -- the node count is expected to differ, and does.
    The full 1,050 records: `python -m experiments.reference --verify --agent alphabeta`."""
    factory, check_nodes = reference.SEARCHES["alphabeta"]
    assert not check_nodes
    checked, wrong = reference.verify(
        reference.load(), agent_factory=factory, max_depth=3, check_nodes=False
    )
    assert checked == 600
    assert wrong == []
