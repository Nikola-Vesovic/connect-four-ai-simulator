"""Phase 9: iterative deepening, and the first budget in this project that is enforced.

What is tested here is different in kind from Phases 7 and 8, and the reason is one
sentence: **this is the first optimisation that is allowed to change the move played.**

Every phase since Phase 6 has been held to "same value, same column, fewer nodes". Move
ordering and the transposition table could both be held to the stronger half because the
root sees empty tables and never probes, so its order is the base order whatever the flags
say. Iterative deepening hands the root the previous iteration's best move, and among
moves of *equal value* the root keeps the first one it saw -- so a different column of the
same value can come back. The contract that survives is the one the reference set has
always enforced and no more: **the value**.

That leaves three things this file has to pin instead:

* the value contract itself, sharpened -- an iteration that completed at depth d must
  return exactly what a direct depth-d search returns;
* that a search which carries nothing between iterations (`NAIVE`) still plays the
  direct search's column, so the move change is attributable to the two switches that
  cause it rather than to iterating;
* the budget, which is new machinery with a classic bug in it -- a half-finished
  iteration must contribute nothing at all.
"""

from __future__ import annotations

import random

import pytest

from agents.alphabeta import _CLOCK_INTERVAL, NegamaxAgent
from core.bitboard import Bitboard
from experiments import reference
from helpers import random_position
from search.deepening import DIRECT, EVEN, ITERATIVE, NAIVE, Deepening
from search.evaluation import DEFAULT
from search.ordering import DYNAMIC
from search.stats import Budget, SearchStats
from search.transposition import NO_TABLE, TABLE

PV_ONLY = Deepening(enabled=True, root_pv=True, share_order=False)
ORDER_ONLY = Deepening(enabled=True, root_pv=False, share_order=True)


def positions(seed: int, count: int, max_plies: int = 26):
    rng = random.Random(seed)
    found = []
    while len(found) < count:
        board = random_position(rng, max_plies)
        if not board.is_terminal():
            found.append(board)
    return found


# --------------------------------------------------------------------------- the spec


def test_the_spec_names_itself_and_refuses_a_parity_it_does_not_know():
    assert DIRECT.name == "direct"
    assert NAIVE.name == "iterative"
    assert ITERATIVE.name == "iterative+pv+order"
    assert EVEN.name == "iterative+pv+order+even"
    with pytest.raises(ValueError):
        Deepening(enabled=True, parity="odd")


def test_the_spec_says_which_settings_can_change_the_move():
    """The boundary Phases 7 and 8 both drew by test, now crossed deliberately.

    Written as a property of the setting rather than as a comment, because it is the one
    fact about this phase that a reader of the results has to have.
    """
    assert not DIRECT.can_change_the_move
    assert not NAIVE.can_change_the_move  # carries nothing: the root order is the base one
    assert PV_ONLY.can_change_the_move
    assert ORDER_ONLY.can_change_the_move
    assert ITERATIVE.can_change_the_move


def test_the_agent_names_the_setting_only_when_it_is_not_the_default():
    assert NegamaxAgent(6).name == "alphabeta-d6"
    assert NegamaxAgent(6, deepening=ITERATIVE).name == "alphabeta-d6-iterative+pv+order"


# ------------------------------------------------------- the value contract, sharpened


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
def test_iterative_deepening_returns_exactly_the_fixed_depth_value(depth):
    """The gate of the phase, in miniature. The full version replays all 1,050 frozen
    records via `python -m experiments.reference --verify --deepening iterative`.

    Note what is *not* asserted: the move. Two correct searches may prefer different
    equally-good columns, which has been the reference set's contract since Phase 5 and is
    the first time it is doing real work.
    """
    direct = NegamaxAgent(depth, evaluation=DEFAULT)
    for spec in (NAIVE, ITERATIVE):
        deepened = NegamaxAgent(depth, evaluation=DEFAULT, deepening=spec)
        for board in positions(seed=901 + depth, count=20):
            assert deepened.choose_move(board)[1].value == direct.choose_move(board)[1].value


def test_every_iteration_on_the_way_returns_its_own_depth_value():
    """Stronger than the gate, and the form the gate should have been stated in.

    BUILD_PLAN said "deeper searches are never weaker", which Phase 5 measured to be
    **false** -- depth 3 scores 6.2 points below depth 2 against greedy, because of the
    odd/even effect. The true property is not about strength at all: every iteration on
    the way to depth d must reproduce the search of that depth exactly, which is checked
    here by capping the search at each depth in turn.
    """
    for board in positions(seed=977, count=12):
        for depth in range(1, 7):
            direct = NegamaxAgent(depth, evaluation=DEFAULT).choose_move(board)[1]
            capped = NegamaxAgent(8, evaluation=DEFAULT, deepening=ITERATIVE).choose_move(
                board, Budget(depth=depth)
            )[1]
            assert capped.depth == depth
            assert capped.value == direct.value


# ----------------------------------------------------- what iterating costs, exactly


@pytest.mark.parametrize("depth", [3, 4, 5, 6, 7, 8])
def test_carrying_nothing_costs_exactly_the_separate_searches(depth):
    """The identity that turns the textbook estimate into something falsifiable.

    With nothing carried between iterations *and the table off*, iterating to depth d must
    visit exactly the nodes of d separate searches -- not approximately, exactly. The
    table has to be off for this: it is shared between iterations by construction, and
    the difference between the two numbers is precisely what that sharing is worth.
    """
    separate = sum(
        NegamaxAgent(k, evaluation=DEFAULT, transposition=NO_TABLE)
        .choose_move(Bitboard())[1].nodes
        for k in range(1, depth + 1)
    )
    iterated = NegamaxAgent(
        depth, evaluation=DEFAULT, transposition=NO_TABLE, deepening=NAIVE
    ).choose_move(Bitboard())[1]
    assert iterated.nodes == separate

    with_table = NegamaxAgent(
        depth, evaluation=DEFAULT, transposition=TABLE, deepening=NAIVE
    ).choose_move(Bitboard())[1]
    assert with_table.nodes <= iterated.nodes


def test_the_same_identity_holds_over_real_positions():
    for board in positions(seed=1013, count=10):
        for depth in (4, 5):
            separate = sum(
                NegamaxAgent(k, evaluation=DEFAULT, transposition=NO_TABLE)
                .choose_move(board)[1].nodes
                for k in range(1, depth + 1)
            )
            iterated = NegamaxAgent(
                depth, evaluation=DEFAULT, transposition=NO_TABLE, deepening=NAIVE
            ).choose_move(board)[1]
            assert iterated.nodes == separate


# --------------------------------------------------- which settings can change a move


def test_carrying_nothing_plays_the_column_a_direct_search_plays():
    """The control for trap (a). If this ever failed, the move change could not be
    attributed to the two switches that are supposed to cause it."""
    for depth in (4, 5, 6):
        direct = NegamaxAgent(depth, evaluation=DEFAULT)
        naive = NegamaxAgent(depth, evaluation=DEFAULT, deepening=NAIVE)
        for board in positions(seed=1031 + depth, count=20):
            assert naive.choose_move(board)[0] == direct.choose_move(board)[0]


def test_iterative_deepening_does_change_which_equally_good_move_is_played():
    """The falsifiable half of the same statement, and it has to be stated positively:
    a claim that something *can* happen is only worth making if an example is produced."""
    direct = NegamaxAgent(6, evaluation=DEFAULT)
    iterative = NegamaxAgent(6, evaluation=DEFAULT, deepening=ITERATIVE)
    differences = 0
    for board in positions(seed=1051, count=60):
        a_move, a = direct.choose_move(board)
        b_move, b = iterative.choose_move(board)
        assert a.value == b.value  # the value is the contract and it never moves
        differences += a_move != b_move
    assert differences > 0


def test_the_root_principal_variation_earns_its_place():
    """`root_pv` is an *efficiency* switch, and Phase 7 established that efficiency faults
    are caught by pinned node counts and by nothing else.

    The pinned empty-board row cannot catch this one: there the previous iteration's best
    move is column 3, which the base order already tries first, so ignoring the principal
    variation changes nothing at all. Written because the mutation check found exactly that
    hole -- `root_pv_ignored` survived every other test in the suite.

    **It is measured over the frozen reference positions and not over a fresh random
    sample, and the first version of this test got that wrong.** The saving is 4-12%, which
    is real but small enough that twenty-five arbitrary positions can invert it: they did,
    by 0.5%, while the same count drawn from the reference set gives 5.8%. Same lesson as
    thesis points P11 and P21 at a third scale -- an aggregate effect needs the sample it
    was measured on.
    """
    boards, seen = [], set()
    for record in reference.load()["records"]:
        if record["moves"] not in seen:
            seen.add(record["moves"])
            boards.append(Bitboard.from_move_string(record["moves"]))

    for depth, count in ((4, 120), (5, 60)):
        sample = boards[:count]
        carried = sum(
            NegamaxAgent(depth, evaluation=DEFAULT, deepening=PV_ONLY)
            .choose_move(board)[1].nodes for board in sample
        )
        dropped = sum(
            NegamaxAgent(depth, evaluation=DEFAULT, deepening=NAIVE)
            .choose_move(board)[1].nodes for board in sample
        )
        assert carried < dropped, (depth, carried, dropped)


def test_the_shared_table_is_not_one_of_the_things_that_can_change_it():
    """The root never probes, so the table cannot reorder it however much it knows.

    Which is why the transposition switch stayed inside the Phase 6 guarantee and this
    phase's two switches do not.
    """
    for depth in (4, 5, 6):
        for board in positions(seed=1091 + depth, count=15):
            with_table = NegamaxAgent(
                depth, evaluation=DEFAULT, transposition=TABLE, deepening=NAIVE
            ).choose_move(board)[0]
            without = NegamaxAgent(
                depth, evaluation=DEFAULT, transposition=NO_TABLE, deepening=NAIVE
            ).choose_move(board)[0]
            assert with_table == without


def test_the_killer_slots_at_the_root_stay_empty_however_many_iterations_run():
    """Structural, and it is why `NAIVE` can be a control at all. A cutoff is only ever
    recorded inside `_negamax`, so ply 0 is never written to -- the history table is the
    only part of the ordering state that reaches the root."""
    agent = NegamaxAgent(7, evaluation=DEFAULT, deepening=ITERATIVE)
    order = agent.ordering.start(7, None)
    agent._search_root(Bitboard(), order, None, 5, -1, SearchStats())
    assert order.killers[0] == [-1, -1]
    assert any(slot != [-1, -1] for slot in order.killers[1:])


# ------------------------------------------------------------------------- the budget


def test_a_legal_move_comes_back_however_small_the_budget():
    """Including budgets that were already spent before the search started. Iteration one
    is uninterruptible for exactly this reason."""
    board = Bitboard.from_move_string("4455")
    for budget in (
        Budget(time=0.0), Budget(time=1e-9), Budget(time=1e-6),
        Budget(nodes=1), Budget(nodes=2), Budget(nodes=0),
        Budget(time=0.0, nodes=0),
    ):
        move, stats = NegamaxAgent(12, deepening=ITERATIVE).choose_move(board, budget)
        assert move in board.legal_moves(), budget
        assert stats.depth >= 1 and stats.iterations >= 1, budget
        assert stats.value is not None, budget


def test_the_time_budget_is_honoured():
    """Not a measurement of the overshoot -- that is `deepening_under_a_clock.csv`, and it
    is reported rather than asserted because a loaded machine is not a failing test. What
    is asserted here is the thing that would be a bug: a search that blows straight
    through its budget because the clock is only consulted between iterations."""
    board = Bitboard.from_move_string("4455")
    for budget in (0.05, 0.1):
        _, stats = NegamaxAgent(20, deepening=ITERATIVE).choose_move(
            board, Budget(time=budget)
        )
        assert stats.elapsed < budget * 2.5 + 0.05, (budget, stats.elapsed)
        assert stats.depth >= 1


def test_a_node_budget_is_enforced_exactly_and_reproducibly():
    """The answer to the trap a clock creates. Every pinned number in this thesis comes
    from a node or depth budget precisely because this holds and a deadline does not."""
    board = Bitboard.from_move_string("44553")
    for limit in (200, 2_000, 20_000):
        runs = [
            NegamaxAgent(20, deepening=ITERATIVE).choose_move(board, Budget(nodes=limit))[1]
            for _ in range(3)
        ]
        assert {(r.nodes, r.depth, r.value, r.move) for r in runs} == {
            (runs[0].nodes, runs[0].depth, runs[0].value, runs[0].move)
        }
        assert runs[0].nodes <= limit


def test_an_aborted_iteration_contributes_nothing_to_the_answer():
    """Trap (g), and the classic bug of this algorithm.

    A budget that expires part way through an iteration leaves a root that has looked at
    some of its moves and not others, and the best of those is not the value of anything.
    The property that says it was thrown away whole: the answer of a budgeted search that
    reached depth k is *identical* to the answer of an unbudgeted search capped at k --
    same column, same value -- because the state the aborted iteration touched came after
    iteration k had already finished.
    """
    checked = aborted = 0
    for board in positions(seed=1109, count=10):
        for limit in (3_000, 30_000):
            _, budgeted = NegamaxAgent(20, deepening=ITERATIVE).choose_move(
                board, Budget(nodes=limit)
            )
            _, settled = NegamaxAgent(budgeted.depth, deepening=ITERATIVE).choose_move(board)
            assert (budgeted.move, budgeted.value) == (settled.move, settled.value)
            assert budgeted.nodes >= settled.nodes
            checked += 1
            aborted += budgeted.aborted
    assert checked == 20
    assert aborted > 0  # otherwise the property was never actually exercised


def test_an_abort_leaves_the_board_exactly_as_it_found_it():
    """`Agent.choose_move` refuses an unbalanced play/undo, so this passing at all is the
    assertion; what the test adds is making sure aborts really happen, deep in the tree,
    rather than tidily between iterations."""
    board = Bitboard.from_move_string("4455")
    before = board.moves_played
    aborts = 0
    for limit in range(_CLOCK_INTERVAL, 20 * _CLOCK_INTERVAL, _CLOCK_INTERVAL):
        _, stats = NegamaxAgent(20, deepening=ITERATIVE).choose_move(
            board, Budget(nodes=limit)
        )
        aborts += stats.aborted
        assert board.moves_played == before
    assert aborts > 0


def test_a_depth_budget_still_caps_an_iterative_search():
    _, stats = NegamaxAgent(10, deepening=ITERATIVE).choose_move(
        Bitboard(), Budget(depth=4)
    )
    assert (stats.depth, stats.depth_cap, stats.iterations) == (4, 4, 4)


def test_being_interruptible_does_not_change_the_tree():
    """The Phase 7 and Phase 8 control, repeated for the third time: a budget large enough
    never to fire must leave the search identical, so that every millisecond of difference
    is bookkeeping and can be quoted as such."""
    for depth in (5, 6, 7):
        free = NegamaxAgent(depth, deepening=ITERATIVE).choose_move(Bitboard())[1]
        checked = NegamaxAgent(depth, deepening=ITERATIVE).choose_move(
            Bitboard(), Budget(nodes=10**9)
        )[1]
        assert (free.nodes, free.leaves, free.cutoffs) == (
            checked.nodes, checked.leaves, checked.cutoffs
        )


# ------------------------------------------------------------------------- the table


def test_the_table_is_shared_between_iterations_and_thrown_away_between_moves():
    """Both halves of the sentence, because only one of them is new.

    Sharing between iterations is what makes the phase work, and it shows up as `NAIVE`
    costing fewer nodes than the separate searches it would otherwise be. Throwing it away
    between moves is Phase 8's finding and is *not* reversed here: two identical searches
    of the same position must cost exactly the same, or a node count stops being a
    property of a position.
    """
    board = Bitboard.from_move_string("4455")
    agent = NegamaxAgent(7, evaluation=DEFAULT, transposition=TABLE, deepening=ITERATIVE)
    first = agent.choose_move(board)[1]
    second = agent.choose_move(board)[1]
    assert (first.nodes, first.value, first.move) == (second.nodes, second.value, second.move)

    untabled = NegamaxAgent(
        7, evaluation=DEFAULT, transposition=NO_TABLE, deepening=ITERATIVE
    ).choose_move(board)[1]
    assert first.nodes < untabled.nodes  # the sharing between iterations is worth something


def test_the_transposition_move_finally_does_the_ordering():
    """Phase 8's failed prediction, and the reason it failed, turned into a test.

    In a fixed-depth search `depth + ply` is constant, so every table hit arrives with
    exactly the depth needed and *answers* the node instead of ordering it -- the table
    move caused 0.63% of cutoffs. Iterating breaks that: the same key now carries
    different depths on different iterations, so a shallow entry can order a deep search.
    """
    board = Bitboard.from_move_string("4455")
    direct = NegamaxAgent(7, evaluation=DEFAULT).choose_move(board)[1]
    iterative = NegamaxAgent(7, evaluation=DEFAULT, deepening=ITERATIVE).choose_move(board)[1]
    assert direct.tt_move_cutoffs / direct.cutoffs < 0.05
    assert iterative.tt_move_cutoffs / iterative.cutoffs > 0.20
    # and the killers give up the share the table move takes
    assert iterative.killer_cutoffs / iterative.cutoffs < direct.killer_cutoffs / direct.cutoffs


# -------------------------------------------------------------------------- parity


def test_even_parity_only_ever_answers_from_an_even_depth():
    """Phase 5's odd/even effect is the reason this switch exists: a clock stops the search
    wherever it got to, so a time-budgeted agent's horizon parity varies from move to move,
    which is a thing no fixed-depth agent in this project has ever done."""
    board = Bitboard.from_move_string("4455")
    for limit in (500, 5_000, 50_000):
        _, stats = NegamaxAgent(20, deepening=EVEN).choose_move(board, Budget(nodes=limit))
        assert stats.depth % 2 == 0
        _, any_depth = NegamaxAgent(20, deepening=ITERATIVE).choose_move(
            board, Budget(nodes=limit)
        )
        assert any_depth.depth >= stats.depth


def test_even_parity_falls_back_to_depth_one_when_nothing_even_has_finished():
    _, stats = NegamaxAgent(20, deepening=EVEN).choose_move(Bitboard(), Budget(nodes=1))
    assert stats.depth == 1 and stats.move in Bitboard().legal_moves()


def test_at_a_fixed_even_depth_the_parity_rule_changes_nothing():
    for depth in (4, 6):
        for board in positions(seed=1153 + depth, count=12):
            plain = NegamaxAgent(depth, evaluation=DEFAULT, deepening=ITERATIVE)
            even = NegamaxAgent(depth, evaluation=DEFAULT, deepening=EVEN)
            assert plain.choose_move(board)[1].value == even.choose_move(board)[1].value
