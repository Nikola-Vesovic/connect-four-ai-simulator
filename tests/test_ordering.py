"""Phase 7: move ordering.

The ordering is the one part of the search that **cannot** change the answer, and the
tests are shaped by that. `search/ordering.py` never sees a score -- it is handed a board
and a ply and returns a list of legal columns -- so soundness is structural here rather
than something to be checked case by case. What is left to test is therefore in three
groups, and they are quite different in character:

* **it changed nothing it was not supposed to change** -- the same value, the same move,
  and with the dynamic parts switched off the same tree Phase 6 measured, node for node;
* **it did change what it was supposed to change** -- materially fewer nodes over real
  positions, with the ordering-quality metrics moving in the same direction;
* **the failure mode is real** -- a bad base order costs a great deal, which is slide
  148's warning turned into a measurement instead of a repetition.

The bookkeeping itself (killer slots, the history table) is unit-tested directly, because
a fault in it is invisible from outside: it would make the search slower and nothing else,
and a slow search still returns the right answer.
"""

from __future__ import annotations

import random

import pytest

from agents.alphabeta import NegamaxAgent
from core.bitboard import Bitboard
from core.rules import COLS
from core.types import Player
from helpers import random_position
from search.ordering import (
    CENTER_FIRST,
    DYNAMIC,
    KILLER_SLOTS,
    LEFT,
    LEFT_TO_RIGHT,
    RANDOM,
    STATIC,
    MoveOrder,
    Ordering,
)
from search.scouting import NO_SCOUT

ALL = (STATIC, DYNAMIC, LEFT, RANDOM)


def positions(seed: int, count: int):
    rng = random.Random(seed)
    made = 0
    while made < count:
        board = random_position(rng)
        if board.is_terminal():
            continue
        made += 1
        yield board


# ------------------------------------------------------------------- the bookkeeping


def test_only_legal_columns_are_ever_offered():
    board = Bitboard.from_move_string("1" * 6 + "7" * 6)  # columns 0 and 6 are full
    for spec in ALL:
        order = spec.start(4, seed=1)
        assert sorted(order.columns(board, 0)) == [1, 2, 3, 4, 5], spec.name


def test_with_both_switches_off_the_order_is_simply_the_base_order():
    """Which is why the static configuration is not a second code path: the same function
    runs over empty tables and a stable sort leaves the base order exactly as it was."""
    board = Bitboard()
    assert STATIC.start(4).columns(board, 0) == list(CENTER_FIRST)
    assert LEFT.start(4).columns(board, 0) == list(LEFT_TO_RIGHT)


def test_a_killer_is_tried_first_next_time_that_ply_is_reached():
    board = Bitboard()
    order = Ordering(killers=True, history=False).start(4)
    assert order.columns(board, 2) == list(CENTER_FIRST)

    order.cutoff(board, 6, ply=2, depth=3)
    assert order.columns(board, 2)[0] == 6
    assert order.promoted == 1
    # A killer learned at one ply says nothing about another one.
    assert order.columns(board, 3) == list(CENTER_FIRST)


def test_the_killer_slots_hold_the_two_most_recent_and_never_the_same_move_twice():
    board = Bitboard()
    order = Ordering(killers=True, history=False).start(4)
    for col in (0, 1, 2):
        order.cutoff(board, col, ply=1, depth=2)
    assert order.killers[1] == [2, 1]  # 0 has been pushed out
    assert order.promoted == 0  # not set until columns() is called
    assert order.columns(board, 1)[:2] == [2, 1]

    order.cutoff(board, 2, ply=1, depth=2)  # already in front: nothing moves
    assert order.killers[1] == [2, 1]
    assert len(order.killers[1]) == KILLER_SLOTS


def test_history_is_kept_per_side_because_a_good_column_is_good_for_somebody():
    empty = Bitboard()  # player ONE to move
    one_played = Bitboard.from_move_string("1")  # player TWO to move

    order = Ordering(killers=False, history=True).start(4)
    order.cutoff(empty, 6, ply=1, depth=3)

    assert order.columns(empty, 1)[0] == 6
    assert order.columns(one_played, 1)[0] == CENTER_FIRST[0]  # the other side learnt nothing


def test_a_refutation_found_higher_up_counts_for_more():
    """`depth * depth`: a cutoff near the root stood for a far larger subtree than one
    found next to the leaves, so it is stronger evidence about the column."""
    board = Bitboard()
    order = Ordering(killers=False, history=True).start(6)
    order.cutoff(board, 0, ply=1, depth=5)  # 25
    for _ in range(4):
        order.cutoff(board, 6, ply=1, depth=2)  # 4 each, 16 in total
    assert order.columns(board, 1)[0] == 0


def test_columns_nothing_is_known_about_keep_the_base_order_among_themselves():
    board = Bitboard()
    order = Ordering(killers=False, history=True).start(4)
    order.cutoff(board, 0, ply=1, depth=3)
    assert order.columns(board, 1) == [0] + [col for col in CENTER_FIRST if col != 0]


def test_an_unknown_base_order_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="Unknown base order"):
        Ordering(base="centre")  # the British spelling, and a plausible typo


def test_every_ordering_names_itself_and_the_name_says_what_is_on():
    assert STATIC.name == "center_first"
    assert DYNAMIC.name == "center_first+killers+history"
    assert Ordering(killers=True, history=False).name == "center_first+killers"
    assert RANDOM.name == "random"
    assert LEFT.name == "left_to_right"


def test_the_agent_says_which_ordering_it_used_unless_it_is_the_default():
    assert NegamaxAgent(6).name == "alphabeta-d6"
    assert NegamaxAgent(6, ordering=Ordering()).name == "alphabeta-d6"  # equal, not identical
    assert NegamaxAgent(6, ordering=STATIC).name == "alphabeta-d6-center_first"
    assert NegamaxAgent(6, ordering=RANDOM).name == "alphabeta-d6-random"


# --------------------------------------------------------- it changed nothing it must not


@pytest.mark.parametrize("ordering", ALL, ids=lambda spec: spec.name)
def test_no_ordering_can_change_the_value(ordering):
    """The whole correctness story of the phase, and it is cheap because the ordering
    module never sees a score. The full version is the 1,050 frozen records replayed by
    `python -m experiments.reference --verify --agent alphabeta`."""
    baseline = NegamaxAgent(5, ordering=STATIC)
    other = NegamaxAgent(5, ordering=ordering, seed=1)
    for board in positions(seed=211, count=25):
        assert other.choose_move(board)[1].value == baseline.choose_move(board)[1].value


def test_killers_and_history_cannot_change_the_move_that_gets_played():
    """Stronger than the contract asks for, and it is the reason every playing-strength
    number from Phases 5 and 6 carries over to Phase 7 without being re-measured.

    It is structural, not luck: at the root the killer slots are empty and the history
    table is all zeroes, so the root tries columns in the base order exactly as Phase 6
    did, and the root replaces its best only on a strict improvement. A random *base*
    order is the one that may legitimately name a different equally-good move, which is
    why it is excluded here and not from the value test above.
    """
    for depth in (2, 4, 6):
        phase6 = NegamaxAgent(depth, ordering=STATIC)
        phase7 = NegamaxAgent(depth, ordering=DYNAMIC)
        for board in positions(seed=223 + depth, count=20):
            assert phase7.choose_move(board)[0] == phase6.choose_move(board)[0]


def test_a_search_does_not_remember_the_previous_one():
    """The tables are per-search and thrown away, so the same position always produces
    the same tree no matter what was searched before it. Reproducibility of every number
    in the thesis rests on this."""
    agent = NegamaxAgent(5)
    board = Bitboard.from_move_string("4455")
    first = agent.choose_move(board)[1]
    for other in positions(seed=227, count=5):
        agent.choose_move(other)
    again = agent.choose_move(board)[1]
    assert (again.nodes, again.cutoffs, again.value) == (first.nodes, first.cutoffs, first.value)


def test_without_pruning_the_ordering_cannot_matter():
    """No cutoffs means nothing to order for, so plain negamax keeps Phase 5's property
    that its node count is independent of the ordering -- and there is nothing for the
    killer table to learn either."""
    for board in positions(seed=229, count=10):
        counts = {
            spec.name: NegamaxAgent(4, pruning=False, ordering=spec, seed=1)
            .choose_move(board)[1]
            .nodes
            for spec in (STATIC, DYNAMIC, LEFT)
        }
        assert len(set(counts.values())) == 1, counts


# ------------------------------------------------------- it did change what it must


def test_over_real_positions_the_learned_ordering_visits_materially_fewer_nodes():
    """The gate of the phase, in miniature. The full version is the 200-position sweep in
    `data/results/ordering_by_position.csv`; this is the part that runs in the suite.

    Deliberately *not* measured on the empty board: that is alpha-beta's best case
    (thesis point P11), centre-first ordering is already close to ideal there, and the
    learned ordering has nothing left to win.

    Pinned at `scouting=NO_SCOUT`, which is the configuration this was measured in.
    NegaScout prunes hard on a null window whatever order the moves arrive in, so it
    takes back part of what the ordering is worth -- 1.60x here instead of 1.66x -- and
    that interaction is a Phase 10 result (`scout_by_ordering.csv`), not a reason to
    relax a Phase 7 number.
    """
    static_total = dynamic_total = 0
    for board in positions(seed=233, count=25):
        _, before = NegamaxAgent(6, ordering=STATIC, scouting=NO_SCOUT).choose_move(board)
        _, after = NegamaxAgent(6, ordering=DYNAMIC, scouting=NO_SCOUT).choose_move(board)
        assert after.value == before.value
        static_total += before.nodes
        dynamic_total += after.nodes
    assert dynamic_total < static_total / 1.5, (static_total, dynamic_total)


def test_ordering_quality_is_reported_and_the_learned_ordering_improves_it():
    """The metric the plan asks for in its own right. A cutoff on the first move tried is
    a node where the ordering was perfect; the share of those is what ordering work is
    trying to raise, and it is a different quantity from the node count."""
    for name, ordering, floor in (("static", STATIC, 0.0), ("dynamic", DYNAMIC, 0.80)):
        cutoffs = first = index_total = 0
        for board in positions(seed=239, count=20):
            _, stats = NegamaxAgent(6, ordering=ordering).choose_move(board)
            assert stats.cutoffs > 0
            cutoffs += stats.cutoffs
            first += stats.first_cutoffs
            index_total += stats.cut_index_total
            assert stats.first_cutoffs <= stats.cutoffs
            assert stats.cut_index_total >= stats.cutoffs  # every rank is at least 1
        rate = first / cutoffs
        assert rate >= floor, f"{name}: first-move cutoff rate {rate:.1%}"
        if ordering is DYNAMIC:
            assert index_total / cutoffs < 1.5


def test_the_killer_table_is_credited_with_the_cutoffs_it_caused():
    """Attribution, not just effect: without this the gain could be a coincidence of
    centre-first happening to agree with the killers."""
    cutoffs = by_killer = 0
    for board in positions(seed=241, count=15):
        _, stats = NegamaxAgent(6, ordering=DYNAMIC).choose_move(board)
        cutoffs += stats.cutoffs
        by_killer += stats.killer_cutoffs
        assert stats.killer_cutoffs <= stats.cutoffs
    assert by_killer / cutoffs > 0.75

    # Nothing to credit when there is no killer table.
    for board in positions(seed=241, count=5):
        assert NegamaxAgent(6, ordering=STATIC).choose_move(board)[1].killer_cutoffs == 0


# ------------------------------------------------------------------ the failure mode


def test_a_bad_base_order_costs_a_great_deal_and_the_learned_ordering_wins_it_back():
    """Slide 148 warns that a poor ordering undoes the benefit. Measured here in both
    directions, because the recovery is the more interesting half: killers and history
    are not merely an addition to a good static order, they substantially *replace* one.

    Measured over this sample at depth 6: left-to-right costs 110,103 nodes against
    centre-first's 42,057 (2.6x), a per-node shuffle 61,835 (1.5x). Adding killers and
    history to the left-to-right order brings it to 28,405 -- below the good static order
    that has no learning in it at all.

    Pinned at `scouting=NO_SCOUT` for the same reason as the test above: what a bad base
    order costs is a smaller number once a null window is doing part of the pruning, and
    that is Phase 10's finding rather than a correction to this one.
    """
    totals = {spec: 0 for spec in (STATIC, DYNAMIC, LEFT, RANDOM)}
    learned = {LEFT: 0, RANDOM: 0}
    for board in positions(seed=251, count=15):
        for spec in totals:
            _, stats = NegamaxAgent(
                6, ordering=spec, scouting=NO_SCOUT, seed=3
            ).choose_move(board)
            totals[spec] += stats.nodes
        for spec in learned:
            with_learning = Ordering(base=spec.base, killers=True, history=True)
            _, stats = NegamaxAgent(
                6, ordering=with_learning, scouting=NO_SCOUT, seed=3
            ).choose_move(board)
            learned[spec] += stats.nodes

    good = totals[STATIC]
    assert totals[LEFT] > good * 2.0  # the warning is real, and not marginal
    assert totals[RANDOM] > good * 1.3
    assert totals[DYNAMIC] < good / 1.5  # what the phase buys on a good base order

    for spec in learned:
        assert learned[spec] < totals[spec] / 2.0, spec.name  # most of the loss is recovered
        assert learned[spec] < good, spec.name  # and it ends up ahead of the static order


def test_a_random_ordering_is_still_reproducible():
    """It is a measurement, so it has to repeat. The shuffle is drawn from a generator
    seeded per search rather than from the agent's own stream, so the same position gives
    the same tree however many searches came before it."""
    board = Bitboard.from_move_string("4455")
    one = NegamaxAgent(6, ordering=RANDOM, seed=17).choose_move(board)[1]
    two = NegamaxAgent(6, ordering=RANDOM, seed=17).choose_move(board)[1]
    assert (one.nodes, one.cutoffs) == (two.nodes, two.cutoffs)
    assert NegamaxAgent(6, ordering=RANDOM, seed=18).choose_move(board)[1].nodes != one.nodes


# ------------------------------------------------------------------------ small print


def test_the_history_table_has_one_row_per_side_and_one_column_per_column():
    order = MoveOrder(DYNAMIC, max_ply=4)
    assert len(order.history) == len(Player)
    assert all(len(row) == COLS for row in order.history)
