"""Phase 8: the transposition table.

The phase splits cleanly into three things that need three different kinds of test, and
keeping them apart is most of the work.

**The key** is a claim about the game, not about the code: `position + mask` names a
Connect Four position uniquely. If that is false, everything above it is unsound in a way
no amount of search testing would reliably reveal, so it is tested directly and
exhaustively-ish -- against the full cell contents, over every position two independent
board implementations can reach.

**The table** is a container. Store, probe, replace, run out of room. Ordinary unit tests,
except for one thing that is not ordinary: the fixed-size table is allowed to *lose*
entries, and the tests say so explicitly rather than pretending both implementations
behave identically.

**The search** is where the table can do damage. Move ordering could only ever make the
search slow (Phase 7), because it never sees a score. A transposition table returns
values, so a mistake in the bound logic changes answers. That is why the gate here is the
same equality Phase 6 used -- the table on and the table off must agree -- and why it is
checked at several depths rather than one.
"""

from __future__ import annotations

import random

import pytest

from agents.alphabeta import NegamaxAgent
from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from core.rules import COLS, ROWS
from core.types import Score
from experiments import reference
from helpers import random_position
from search.evaluation import DEFAULT
from search.ordering import DYNAMIC, STATIC
from search.transposition import (
    ARRAY,
    EXACT,
    LOWER,
    NO_TABLE,
    PERSISTENT,
    TABLE,
    UPPER,
    WRITE_ONLY,
    ArrayTable,
    DictTable,
    Transposition,
    from_entry,
    to_entry,
)


def positions(seed: int, count: int):
    rng = random.Random(seed)
    made = 0
    while made < count:
        board = random_position(rng)
        if board.is_terminal():
            continue
        made += 1
        yield board


def contents(board) -> tuple:
    """Everything a search result may depend on, read the slowest, most obvious way."""
    return (
        int(board.player_to_move),
        tuple(board.cell(row, col) for row in range(ROWS) for col in range(COLS)),
    )


# ------------------------------------------------------------------------------ the key


def test_the_key_is_collision_free_over_every_position_two_boards_can_reach():
    """The claim the whole phase rests on. A hash table that collides returns an answer
    belonging to a different position; this key cannot, and that is a property of the
    encoding rather than of the table.

    Per column the contribution is `own_discs + 2**height - 1`, which lands in
    `[2**h - 1, 2**(h+1) - 2]`. Those intervals are disjoint for every height, so the
    contribution determines the height and the discs, and the widest of them still fits
    in the seven bits the column owns. The test is the empirical half of that argument.
    """
    rng = random.Random(2026)
    seen: dict[int, tuple] = {}
    collisions = 0
    for _ in range(400):
        board = Bitboard()
        while not board.is_terminal():
            board.play(rng.choice(board.legal_moves()))
            here = contents(board)
            if seen.setdefault(board.key(), here) != here:
                collisions += 1
    assert collisions == 0
    assert len(seen) > 7_000, f"only {len(seen)} distinct positions tested"  # 7,382, measured


def test_both_board_implementations_name_a_position_the_same_way():
    rng = random.Random(7)
    for _ in range(200):
        fast, slow = Bitboard(), NaiveBoard()
        while not fast.is_terminal():
            col = rng.choice(fast.legal_moves())
            fast.play(col)
            slow.play(col)
            assert fast.key() == slow.key(), fast.history


def test_the_key_says_whose_turn_it_is():
    """Two positions with the same discs but different sides to move are different
    positions, and a table that confused them would hand a search the opponent's value."""
    one = Bitboard.from_move_string("4455")  # four discs, player one to move
    other = Bitboard.from_move_string("44556")
    assert one.key() != other.key()

    # And the same arrangement reached two ways really is the same key.
    assert (
        Bitboard.from_move_string("415").key() == Bitboard.from_move_string("514").key()
    )


def test_the_empty_board_has_key_zero_which_is_why_slots_store_the_key_offset_by_one():
    assert Bitboard().key() == 0
    table = ArrayTable(64)
    assert table.probe(0) is None
    table.store(0, 3, EXACT, 17, 4)
    assert table.probe(0) == (3, EXACT, 17, 4)


# -------------------------------------------------------------------- mate score arithmetic


@pytest.mark.parametrize("ply", [0, 1, 5, 20, 41])
def test_a_value_survives_a_round_trip_at_the_ply_it_was_written_at(ply):
    for value in (
        Score.DRAW, 17, -240, 1781, -1781,
        Score.win_in(ply), Score.loss_in(ply), Score.win_in(41), Score.loss_in(41),
    ):
        assert from_entry(to_entry(value, ply), ply) == value


def test_a_mate_read_back_at_another_ply_keeps_its_distance_from_the_node():
    """The trap. `Score.win_in(n)` counts plies from the ROOT, so the same mate is a
    different number at every depth in the tree. Storing it raw and reading it back
    somewhere else would move the win closer or further away by the difference.

    What is stored is therefore the distance from the *node*, which is what a position
    actually owns. A win two plies below a node is a win two plies below that node
    wherever the node happens to sit.
    """
    stored = to_entry(Score.win_in(7), ply=5)  # a win two plies below a node at ply 5
    assert from_entry(stored, ply=5) == Score.win_in(7)
    assert from_entry(stored, ply=2) == Score.win_in(4)  # same two plies, different node
    assert from_entry(stored, ply=0) == Score.win_in(2)

    losing = to_entry(Score.loss_in(9), ply=6)
    assert from_entry(losing, ply=1) == Score.loss_in(4)


def test_a_heuristic_value_is_stored_untouched():
    """Only terminal scores carry a distance. A static judgement about a quiet position
    is the same number wherever the position is met, and adjusting it would corrupt it."""
    for value in range(-1782, 1783, 61):
        assert to_entry(value, 11) == value
        assert from_entry(value, 11) == value


# --------------------------------------------------------------------------- the container


@pytest.mark.parametrize("build", [lambda: DictTable(1 << 12), lambda: ArrayTable(1 << 12)])
def test_what_goes_in_comes_back_out(build):
    table = build()
    table.store(key=12345, depth=7, flag=LOWER, value=-9993, move=6)
    assert table.probe(12345) == (7, LOWER, -9993, 6)
    assert table.probe(12346) is None
    assert table.probes == 2 and table.hits == 1 and table.stores == 1
    assert table.entries == 1


@pytest.mark.parametrize("build", [lambda: DictTable(1 << 12), lambda: ArrayTable(1 << 12)])
@pytest.mark.parametrize("flag", [EXACT, LOWER, UPPER])
def test_every_field_survives_storage_over_the_whole_range_it_can_hold(build, flag):
    """The fixed-size table packs an entry into one machine word, so this is really a
    test of the packing -- and it is run against the dictionary too, so that the two
    cannot drift into meaning different things."""
    table = build()
    for value in (Score.LOSS, Score.MAX_LOSS, -1782, 0, 1782, Score.MIN_WIN, Score.WIN):
        for depth in (0, 1, 13, 42):
            for move in range(COLS):
                key = value * 1_000_003 + depth * 71 + move
                table.store(key=abs(key), depth=depth, flag=flag, value=value, move=move)
                assert table.probe(abs(key)) == (depth, flag, value, move)


def test_the_dictionary_stops_growing_when_it_is_full_rather_than_evicting():
    """Trap (g): a deep search can otherwise fill memory with millions of entries. An
    exact table has no cheap way to choose a victim, so the cap is a stop, not a policy --
    and a refused store is counted rather than silent."""
    table = DictTable(10)
    for key in range(25):
        table.store(key=key, depth=4, flag=EXACT, value=key, move=0)
    assert table.entries == 10
    assert table.overflows == 15
    assert table.probe(3) is not None and table.probe(20) is None


def test_the_fixed_size_table_never_exceeds_its_capacity():
    table = ArrayTable(64)
    for key in range(5_000):
        table.store(key=key, depth=4, flag=EXACT, value=1, move=0)
    assert table.entries <= 64
    assert table.collisions > 0  # it really did have to throw things away


def test_replacement_prefers_the_deeper_result():
    """Depth-preferred: a deep entry cost more to compute and stands for a larger subtree.

    Note what this is *not* worth inside a single fixed-depth search. Every occurrence of
    a position in one search carries the same remaining depth, because a move always adds
    exactly one disc, so the policy only ever chooses between equals there. It is the
    policy that matters once a table outlives one search.
    """
    table = ArrayTable(1)  # one slot: every key collides, deliberately
    table.store(key=1, depth=9, flag=EXACT, value=100, move=1)
    table.store(key=2, depth=3, flag=EXACT, value=200, move=2)
    assert table.probe(1) == (9, EXACT, 100, 1), "a shallow result evicted a deep one"

    table.store(key=3, depth=9, flag=EXACT, value=300, move=3)
    assert table.probe(3) == (9, EXACT, 300, 3), "equal depth should go to the newcomer"


def test_the_counters_count_what_they_say_they_do():
    table = DictTable(1 << 8)
    table.store(key=5, depth=4, flag=EXACT, value=1, move=0)
    table.store(key=5, depth=4, flag=LOWER, value=2, move=1)  # same position again
    table.probe(5)
    table.probe(6)
    assert (table.stores, table.rewrites, table.probes, table.hits) == (2, 1, 2, 1)
    assert table.entries == 1 and table.peak == 1
    table.reset_counters()
    assert (table.stores, table.probes, table.hits) == (0, 0, 0)
    assert table.entries == 1, "clearing the counters must not clear the table"
    table.clear()
    assert table.entries == 0


def test_a_table_of_an_impossible_size_is_refused():
    with pytest.raises(ValueError, match="power of two"):
        ArrayTable(1000)
    with pytest.raises(ValueError, match="at least one entry"):
        Transposition(capacity=0)
    with pytest.raises(ValueError, match="Unknown table kind"):
        Transposition(kind="btree")


def test_the_name_says_which_table_it_is():
    assert TABLE.name == "dict"
    assert NO_TABLE.name == "no_table"
    assert ARRAY.name == "array"
    assert WRITE_ONLY.name == "dict+write_only"
    assert PERSISTENT.name == "dict+persistent"
    assert Transposition(kind="array", capacity=1 << 12).name == "array4k"
    assert NegamaxAgent(8).name == "alphabeta-d8"
    assert NegamaxAgent(8, transposition=NO_TABLE).name == "alphabeta-d8-no_table"


# ---------------------------------------------------------------- the ordering it feeds


def test_the_table_move_is_tried_before_the_killers():
    """The slot Phase 7 left empty, now filled. A killer is a refutation borrowed from a
    sibling position; a table move was computed for this exact one, so it goes first."""
    order = DYNAMIC.start(6)
    board = Bitboard.from_move_string("44")
    order.killers[2] = [5, 6]

    assert order.columns(board, 2)[:2] == [5, 6]
    assert order.tt_promoted == 0 and order.promoted == 2

    with_table = order.columns(board, 2, tt_move=1)
    assert with_table[:3] == [1, 5, 6]
    assert order.tt_promoted == 1 and order.promoted == 3
    assert sorted(with_table) == board.legal_moves()


def test_a_table_move_is_promoted_even_with_the_killers_switched_off():
    """They are separate optimisations and Phase 14's ablation matrix needs them to be.
    The table move arrives from the transposition switch, not from the ordering one."""
    order = STATIC.start(4)
    board = Bitboard.from_move_string("44")
    assert order.columns(board, 1, tt_move=6)[0] == 6
    assert order.tt_promoted == 1


def test_an_unplayable_table_move_is_ignored():
    """A table move for a column that has since filled up would make `Board.play` raise.
    The same guard the killers have always had.

    The witness fills columns 1 and 2 while leaving the game alive: each is stacked
    alternately, so neither has four of a colour, and two columns cannot hold a
    horizontal or a diagonal line at all.
    """
    board = Bitboard.from_move_string("122112211221")
    assert not board.is_terminal() and not board.can_play(0)

    order = DYNAMIC.start(4)
    columns = order.columns(board, 1, tt_move=0)
    assert columns == [col for col in columns if board.can_play(col)]
    assert sorted(columns) == board.legal_moves()
    assert order.tt_promoted == 0


# ------------------------------------------------------------------- the gate: same answer


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_the_table_never_changes_the_value(depth):
    """Design principle 3, and the one place in this project where it is genuinely at
    risk. An entry is not always a value -- a node that cut off stored a bound -- so using
    one outside the window it is valid for would return a number no search would."""
    off = NegamaxAgent(depth, transposition=NO_TABLE, evaluation=DEFAULT)
    for spec in (TABLE, ARRAY, Transposition(kind="array", capacity=64)):
        on = NegamaxAgent(depth, transposition=spec, evaluation=DEFAULT)
        for board in positions(seed=200 + depth, count=20):
            assert on.choose_move(board)[1].value == off.choose_move(board)[1].value, spec.name


def test_the_table_never_changes_the_move_that_gets_played():
    """Stronger than the contract requires, and the same property Phases 6 and 7 pin.

    It holds for a precise reason: the root needs a move rather than a value, so it never
    probes, which leaves its move order the base order. Thesis point P18 -- the
    OPTIMIZACIJE checkboxes change what a move costs and never what it is.
    """
    for depth in (3, 4, 5):
        off = NegamaxAgent(depth, transposition=NO_TABLE)
        on = NegamaxAgent(depth, transposition=TABLE)
        for board in positions(seed=210 + depth, count=20):
            assert on.choose_move(board)[0] == off.choose_move(board)[0]


def test_the_table_reproduces_the_frozen_reference_values():
    """The Phase 5 net, still holding. Full run:
    `python -m experiments.reference --verify --agent alphabeta --table dict`."""
    factory = reference.SEARCHES["alphabeta"][0]
    checked, wrong = reference.verify(reference.load(), agent_factory=factory, max_depth=3)
    assert checked == 600
    assert wrong == []


def test_switching_the_table_off_reproduces_phase_sevens_search_exactly():
    """"Off" has to mean *off*, not "a table that happens not to help". With `NO_TABLE`
    the agent never calls `board.key()` and never allocates a table at all."""
    for board in positions(seed=221, count=10):
        _, stats = NegamaxAgent(5, transposition=NO_TABLE).choose_move(board)
        assert stats.tt_probes == 0 and stats.tt_stores == 0 and stats.tt_capacity == 0


def test_a_table_that_is_kept_but_never_read_searches_exactly_the_same_tree():
    """The controlled experiment that separates what the table costs from what it saves.

    `WRITE_ONLY` probes and stores exactly as the real thing does and then ignores what it
    found, so the tree it searches is provably the Phase 7 tree -- node for node, leaf for
    leaf. Every millisecond of difference is bookkeeping. Measured at 1-3% (BUILD_PLAN).
    """
    for board in positions(seed=223, count=10):
        _, plain = NegamaxAgent(6, transposition=NO_TABLE).choose_move(board)
        _, kept = NegamaxAgent(6, transposition=WRITE_ONLY).choose_move(board)
        assert (kept.nodes, kept.leaves, kept.cutoffs) == (plain.nodes, plain.leaves, plain.cutoffs)
        assert kept.value == plain.value
        assert kept.tt_stores > 0 and kept.tt_returns == 0


def test_the_table_actually_removes_nodes():
    """The other half: it must not merely be harmless. Measured over the 200 reference
    positions at 1.06x at depth 4 rising to 1.51x at depth 7."""
    total_off = total_on = 0
    for board in positions(seed=227, count=15):
        total_off += NegamaxAgent(7, transposition=NO_TABLE).choose_move(board)[1].nodes
        total_on += NegamaxAgent(7, transposition=TABLE).choose_move(board)[1].nodes
    assert total_on < total_off * 0.85, f"{total_off} -> {total_on}"


def test_it_helps_most_where_move_ordering_helped_least():
    """The empty board, which is the whole shape of the phase in one assertion.

    Phase 7 *lost* here -- a learned ordering is 0.77x at depth 8 -- because centre-first
    is already near perfect on a symmetric position. Symmetry is exactly what makes the
    same position reachable by many move orders, so it is where the table has most to
    find, and the prediction was written down before it was run.
    """
    _, off = NegamaxAgent(9, transposition=NO_TABLE, evaluation=DEFAULT).choose_move(Bitboard())
    _, on = NegamaxAgent(9, transposition=TABLE, evaluation=DEFAULT).choose_move(Bitboard())
    assert on.value == off.value
    assert on.nodes < off.nodes / 2  # measured: 53,631 -> 26,102


# --------------------------------------------------------------------------- persistence


def test_by_default_a_search_is_a_function_of_the_position_and_nothing_else():
    """The same reason `MoveOrder` is per-search. If the table survived, the tree a
    position produces would depend on which positions were searched before it, and every
    node count in this thesis would stop being reproducible."""
    agent = NegamaxAgent(6, transposition=TABLE)
    board = Bitboard.from_move_string("4455")
    first = agent.choose_move(board)[1]
    for _ in range(3):
        agent.choose_move(Bitboard.from_move_string("1213"))
    again = agent.choose_move(board)[1]
    assert (again.nodes, again.value, again.tt_probes) == (first.nodes, first.value, first.tt_probes)


def test_a_persistent_table_is_warm_and_that_is_the_point_and_the_price():
    """Measured rather than adopted. It is worth 1.755x over complete games against the
    1.279x a per-search table manages, and it pays for that by making the search depend on
    its own history -- the same position costs a different number of nodes the second
    time. Both halves are asserted here."""
    agent = NegamaxAgent(6, transposition=PERSISTENT)
    board = Bitboard.from_move_string("4455")
    first = agent.choose_move(board)[1]
    again = agent.choose_move(board)[1]
    assert again.value == first.value
    assert again.nodes < first.nodes
    assert again.tt_hit_rate > first.tt_hit_rate

    agent.new_game()
    fresh = agent.choose_move(board)[1]
    assert fresh.nodes == first.nodes, "a new game has to forget the previous one"


def test_a_persistent_table_can_change_the_answer_and_that_is_why_it_is_not_the_default():
    """The decisive measurement behind the per-search default, and it is sharper than the
    reproducibility argument that motivated it.

    Within one search a position's key fixes its ply -- every move adds exactly one disc --
    and `depth + ply` is constant, so a stored entry always has *exactly* the depth the
    reader needs. Across searches from unrelated roots that stops being true: the same
    position sits at a different ply, so an entry computed with more depth left than the
    reader needs passes the `>=` test and answers with a value from a *deeper* search than
    the one being run. It is not an unsound value. It is the wrong question's answer, and
    design principle 3 says an optimisation must not change the answer even by improving
    it.

    The witness is the exact shape that triggers it, which is also the shape
    `experiments.reference` has: one agent asked about many unrelated positions. Feeding
    the same sixty positions to it in two different orders produces two different sets of
    answers, which is as direct a demonstration as the property allows -- a search that
    depends on what it was asked before is not a function of the position.
    """
    records = [r for r in reference.load()["records"] if r["depth"] == 5][:60]

    def replay(spec, order_seed: int) -> dict[str, int]:
        rows = list(records)
        random.Random(order_seed).shuffle(rows)
        agent = NegamaxAgent(5, evaluation=DEFAULT, transposition=spec)
        return {
            row["moves"]: agent.choose_move(Bitboard.from_move_string(row["moves"]))[1].value
            for row in rows
        }

    truth = {row["moves"]: row["value"] for row in records}
    assert replay(TABLE, 1) == truth
    assert replay(TABLE, 2) == truth

    warm_one, warm_two = replay(PERSISTENT, 1), replay(PERSISTENT, 2)
    assert warm_one != warm_two, "the hazard this default exists to avoid did not fire"
    assert warm_one != truth or warm_two != truth


def test_a_persistent_table_still_cannot_change_the_move_inside_one_game():
    """Even warm, the root does not probe, so the base order at the root is untouched --
    and inside a game the roots advance one ply at a time, so a stored entry never has
    enough depth left to answer the deeper question the next search asks. Confirmed over
    ten complete games in `transposition_by_game.csv`, where every table setting,
    persistent included, played the identical 354 moves.

    The qualification matters: the previous test shows this stops holding the moment the
    agent is asked about unrelated positions.
    """
    warm = NegamaxAgent(5, transposition=PERSISTENT)
    cold = NegamaxAgent(5, transposition=NO_TABLE)
    board = Bitboard()
    while not board.is_terminal():
        assert warm.choose_move(board)[0] == cold.choose_move(board)[0]
        board.play(cold.choose_move(board)[0])


# ------------------------------------------------------------------ the numbers it reports


def test_the_statistics_are_reported_and_add_up():
    _, stats = NegamaxAgent(7, transposition=TABLE, evaluation=DEFAULT).choose_move(Bitboard())
    assert stats.tt_probes > 0
    assert stats.tt_hits <= stats.tt_probes
    assert stats.tt_returns <= stats.tt_hits
    assert stats.tt_stores + stats.tt_returns <= stats.tt_probes
    assert stats.tt_entries == stats.tt_stores - stats.tt_rewrites - stats.tt_refused
    assert stats.tt_peak == stats.tt_entries  # a per-search table only ever grows
    assert 0 < stats.tt_hit_rate < 1
    assert stats.tt_fill == stats.tt_entries / stats.tt_capacity


def test_a_hit_is_not_the_same_event_as_a_saving():
    """The distinction the panel line and the CSVs are built on. A probe that finds the
    position has not necessarily saved anything: the entry may be a bound that does not
    settle this window, in which case all it supplies is a move to try first."""
    _, stats = NegamaxAgent(8, transposition=TABLE, evaluation=DEFAULT).choose_move(Bitboard())
    assert stats.tt_returns < stats.tt_hits
    assert stats.tt_return_rate < stats.tt_hit_rate


def test_the_ordering_metrics_still_attribute_a_cutoff_to_exactly_one_source():
    _, stats = NegamaxAgent(8, transposition=TABLE, evaluation=DEFAULT).choose_move(Bitboard())
    assert stats.killer_cutoffs + stats.tt_move_cutoffs <= stats.cutoffs
    assert stats.first_cutoffs <= stats.cutoffs
