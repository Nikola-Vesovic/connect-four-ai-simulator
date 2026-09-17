"""Phase 10: NegaScout, the null window and the re-search.

Three groups, and the split is the phase.

**What must not change.** The value, over the frozen reference set and over random
positions; the column played, which puts this phase back in the category of Phases 7 and
8 after Phase 9 left it; and the search itself when the switch is off, which has to be the
Phase 9 search node for node rather than approximately.

**What the algorithm is betting on.** The share of null-window searches that turn out to
have been wrong and have to be run again. It is reported rather than inferred, and there
is a test that would notice if the counter merely claimed a good number -- the same role
Phase 7's `quality_always_first` mutant gave the ordering-quality metric.

**The degenerate depths.** A depth-1 node's children are horizon nodes that return a
static evaluation whatever window they are handed, so a null window there can save nothing
and every re-search is pure loss. That is an argument until it is an identity, and at
depth 1 it is one: every re-search costs exactly one extra node, so the node count is the
unscouted count plus the number of re-searches, to the node.
"""

from __future__ import annotations

import random

import pytest

from agents.alphabeta import NegamaxAgent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from experiments import reference
from helpers import random_position
from search.deepening import ITERATIVE
from search.evaluation import DEFAULT, ZERO
from search.ordering import DYNAMIC, LEFT, RANDOM, STATIC
from search.scouting import DEEP, NO_SCOUT, SCOUT, WIDE, Scout
from search.transposition import EXACT, NO_TABLE, TABLE

EVERY_SPEC = (NO_SCOUT, SCOUT, WIDE, DEEP)


def positions(seed: int, count: int, max_plies: int = 22):
    rng = random.Random(seed)
    for _ in range(count):
        board = random_position(rng, max_plies=max_plies)
        if not board.is_terminal():
            yield board


# --------------------------------------------------------------- the spec itself


def test_the_name_says_what_the_search_is_doing():
    assert NO_SCOUT.name == "no_scout"
    assert SCOUT.name == "negascout"
    assert WIDE.name == "negascout+wide"
    assert DEEP.name == "negascout+from3"
    assert NegamaxAgent(5).name == "alphabeta-d5"  # the default, so no suffix
    assert NegamaxAgent(5, scouting=NO_SCOUT).name == "alphabeta-d5-no_scout"
    assert NegamaxAgent(5, scouting=WIDE).name == "alphabeta-d5-negascout+wide"


def test_a_search_cannot_be_asked_to_scout_at_depth_zero():
    with pytest.raises(ValueError):
        Scout(min_depth=0)


def test_min_depth_decides_which_nodes_may_scout():
    assert SCOUT.scouts(1) and SCOUT.scouts(9)
    assert not DEEP.scouts(1) and not DEEP.scouts(2) and DEEP.scouts(3)
    assert not NO_SCOUT.scouts(9)


# ------------------------------------------------------ "off" is not a second code path


def test_switching_scouting_off_leaves_the_phase_nine_search_untouched():
    """The window handed to a child is one expression -- `alpha + 1` when the move is
    being scouted, `beta` otherwise -- so with the switch off it is `beta` at every move
    and there is no second implementation to drift apart from the first. Pinned to the
    node, not approximately, exactly as Phase 6 pinned `pruning=False` and Phase 7 pinned
    `Ordering(off)`.
    """
    for board in positions(seed=911, count=12):
        _, off = NegamaxAgent(6, scouting=NO_SCOUT).choose_move(board)
        _, never = NegamaxAgent(6, scouting=Scout(min_depth=99)).choose_move(board)
        assert (never.nodes, never.leaves, never.cutoffs) == (off.nodes, off.leaves, off.cutoffs)
        assert never.scout_searches == 0 and never.researches == 0
        assert off.scout_searches == 0


def test_a_null_window_needs_alpha_beta_and_is_ignored_without_it():
    """A null window is a *pruning* window: with pruning off `alpha` is never raised, so
    `alpha + 1` would sit below the whole score scale and mean nothing. The agent ignores
    the setting entirely there, which is also what keeps the `--verify --agent negamax`
    node-count gate -- pinned to `pruning=False` -- untouched by this phase.
    """
    for board in positions(seed=913, count=6, max_plies=26):
        _, plain = NegamaxAgent(4, pruning=False, transposition=NO_TABLE,
                                scouting=NO_SCOUT).choose_move(board)
        _, scouted = NegamaxAgent(4, pruning=False, transposition=NO_TABLE,
                                  scouting=SCOUT).choose_move(board)
        assert (scouted.nodes, scouted.leaves, scouted.value) == (
            plain.nodes, plain.leaves, plain.value
        )
        assert scouted.scout_searches == 0


# --------------------------------------------------------------------- it changes nothing


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_every_scouting_setting_reproduces_the_reference_values(depth):
    """The gate of the phase, in the suite. The full version replays all 1,050 records
    through `experiments.reference --verify --agent alphabeta --scout ...`."""
    data = reference.load()
    for spec in EVERY_SPEC:
        checked, wrong = reference.verify(
            data,
            agent_factory=lambda d, spec=spec: NegamaxAgent(d, pruning=True, scouting=spec),
            max_depth=depth,
        )
        assert wrong == [], spec.name
        assert checked == 200 * depth


def test_negascout_plays_the_column_alpha_beta_plays():
    """Phase 9 was allowed to name a different one of several equally good moves; this one
    is not, and the argument is exact rather than statistical. At the root the first move
    is searched with the full window and fixes `alpha` at the exact value; a later move
    scouted against that `alpha` and failing low comes back at most `alpha`, and the root
    replaces only on a **strict** improvement. So an equal move can never displace the one
    already held, and a genuinely better one is re-searched with a real window first.
    """
    compared = 0
    sample = list(positions(seed=917, count=40))
    for board in sample:
        baseline, base_stats = NegamaxAgent(5, scouting=NO_SCOUT).choose_move(board)
        for spec in EVERY_SPEC[1:]:
            move, stats = NegamaxAgent(5, scouting=spec).choose_move(board)
            assert (move, stats.value) == (baseline, base_stats.value), spec.name
            compared += 1
    # The floor is the measured yield of the generator, not a round number: it drops the
    # terminal positions, so 40 draws give 31 usable boards.
    assert compared == 3 * len(sample) == 93


def test_the_two_re_search_conventions_agree_on_every_value():
    """Trap (a), as a control rather than an argument. `negascout` re-searches from the
    bound the scout found, `negascout+wide` throws it away and re-searches from alpha.
    Fail-soft is what makes the first one safe -- a scout that fails high returns a lower
    bound on the move's true value -- and if that were wrong, this is where it would show.
    """
    differed = 0
    for board in positions(seed=919, count=20):
        _, narrow = NegamaxAgent(6, scouting=SCOUT).choose_move(board)
        _, wide = NegamaxAgent(6, scouting=WIDE).choose_move(board)
        assert (narrow.value, narrow.move) == (wide.value, wide.move)
        differed += narrow.nodes != wide.nodes
    # ...and they are genuinely different searches, or the control would be vacuous.
    assert differed > 0


def test_scouting_does_not_change_a_game():
    """Phase 9 broke the property that every optimisation combination plays the identical
    game; with the deepening switch off, this phase restores it."""
    played = {}
    for spec in EVERY_SPEC:
        agent = NegamaxAgent(4, scouting=spec, seed=5)
        board = Bitboard()
        while not board.is_terminal():
            board.play(agent.choose_move(board)[0])
        played[spec.name] = "".join(str(col + 1) for col in board.history)
    assert len(set(played.values())) == 1, played


def test_the_value_survives_a_bad_ordering_and_no_table():
    """Slide 148 says a bad ordering costs performance. It must not cost correctness, and
    the combinations where a re-search is most likely are exactly the ones to check."""
    for board in positions(seed=923, count=10):
        expected = MinimaxAgent(4).choose_move(board)[1].value
        for ordering in (DYNAMIC, STATIC, LEFT, RANDOM):
            for table in (TABLE, NO_TABLE):
                _, stats = NegamaxAgent(
                    4, ordering=ordering, transposition=table, scouting=SCOUT, seed=7
                ).choose_move(board)
                assert stats.value == expected, (ordering.name, table.name)


# ------------------------------------------------------------------ the degenerate depths


def test_at_depth_one_a_re_search_costs_exactly_one_node_and_saves_nothing():
    """The sharpest thing in the phase, because it is an identity rather than a ratio.

    The children of a depth-1 node are horizon nodes: they return the static evaluation
    whatever window they are given, so a narrower window cannot cut anything below them.
    Every re-search therefore visits exactly one extra position, and the scouted node
    count is the unscouted one plus the number of re-searches, to the node. It is also
    the only place in this project where an optimisation is provably unable to win.
    """
    total_extra = 0
    for board in positions(seed=929, count=40):
        _, off = NegamaxAgent(1, scouting=NO_SCOUT).choose_move(board)
        _, on = NegamaxAgent(1, scouting=SCOUT).choose_move(board)
        assert on.value == off.value
        assert on.nodes == off.nodes + on.researches
        assert on.leaves == off.leaves + on.researches
        assert on.nodes >= off.nodes  # it can never be cheaper here
        total_extra += on.researches
    assert total_extra > 0, "no re-search happened, so the identity was never exercised"


def test_leaving_the_shallow_depths_alone_is_what_min_depth_is_for():
    """`from3` scouts only where a null window has something to cut. At depth 2 that means
    it does not scout at all, so it must be the unscouted search exactly."""
    for board in positions(seed=931, count=12):
        _, off = NegamaxAgent(2, scouting=NO_SCOUT).choose_move(board)
        _, deep = NegamaxAgent(2, scouting=DEEP).choose_move(board)
        assert (deep.nodes, deep.cutoffs, deep.value) == (off.nodes, off.cutoffs, off.value)
        assert deep.scout_searches == 0


# ------------------------------------------------------------------------ what it buys


def test_over_real_positions_negascout_visits_materially_fewer_nodes():
    """The gate of the phase in miniature; the full version is the 200-position sweep in
    `data/results/scout_by_position.csv`. Depth 6 rather than depth 1, because the whole
    point of the algorithm is that the saving grows with how much tree there is below the
    narrowed window."""
    off_total = on_total = 0
    for board in positions(seed=937, count=25):
        _, off = NegamaxAgent(6, scouting=NO_SCOUT).choose_move(board)
        _, on = NegamaxAgent(6, scouting=SCOUT).choose_move(board)
        assert on.value == off.value
        off_total += off.nodes
        on_total += on.nodes
    assert on_total < off_total / 1.10, (off_total, on_total)


def test_the_bet_is_reported_and_the_counters_are_consistent():
    """The re-search rate is the quantity the algorithm is betting on, so it is a
    measurement and not decoration -- the same argument Phase 7 made for the first-move
    cutoff rate. A counter that always reported a perfect zero would fail here.

    The three totals are pinned exactly, the way the empty-board node counts are, because
    they are what the re-search *policy* looks like from outside. A re-search that fires
    when the scout already reached beta takes 139 to 144; one that fires on a value merely
    equal to alpha takes it to 227; one that forgets to charge itself a node leaves the
    counters alone and drops the node total to 16,837. None of those is a wrong answer and
    none of them was caught by anything else.
    """
    scouts = researches = nodes = 0
    for board in positions(seed=941, count=15):
        _, stats = NegamaxAgent(6, scouting=SCOUT).choose_move(board)
        assert stats.scout_searches > 0
        assert 0 <= stats.researches <= stats.scout_searches
        assert stats.research_rate == pytest.approx(
            stats.researches / stats.scout_searches
        )
        scouts += stats.scout_searches
        researches += stats.researches
        nodes += stats.nodes
    assert (scouts, researches, nodes) == (1035, 139, 16971)
    assert researches / scouts < 0.35  # the bet: wrong guesses are the minority
    assert NegamaxAgent(4, scouting=NO_SCOUT).choose_move(
        Bitboard()
    )[1].research_rate == 0.0


def test_a_better_first_move_makes_the_re_search_rarer_and_the_saving_smaller():
    """Slide 148 says performance depends heavily on the move ordering, and the obvious
    reading -- a better first move means fewer wrong guesses, so NegaScout must win more
    -- is half right and the wrong half is the interesting one.

    Phase 9 measured that iterative deepening raises the transposition move's share of
    cutoffs from under 1% to a quarter or more, i.e. a much better first move. Measured
    here, the first half holds: the re-search rate falls from about 15% to about 9%, and
    that is what is asserted, because the margin is large. The second half does **not**:
    over the 200 reference positions the node gain *falls* with it, from 1.117x to 1.040x
    at depth 6 and from 1.303x to 1.061x at depth 7 (`scout_under_deepening.csv`).
    Alpha-beta with a first move that good is already close to its minimal tree, so there
    is less left for a narrow window to remove. The two optimisations eat the same slack.

    That ratio is deliberately not asserted here. An eleven-board sample inverts it at
    depth 6 (1.027x against 1.040x) while the 200-position sweep says the opposite --
    thesis point P11's sampling lesson again, and the first version of this test asserted
    the inverted direction and passed.
    """
    rates = {}
    for deepening in (None, ITERATIVE):
        kwargs = {} if deepening is None else {"deepening": deepening}
        scouts = researches = 0
        for board in positions(seed=947, count=12):
            _, stats = NegamaxAgent(6, scouting=SCOUT, **kwargs).choose_move(board)
            scouts += stats.scout_searches
            researches += stats.researches
        rates[deepening is not None] = researches / scouts
    assert rates[True] < rates[False] / 1.5, rates


def test_a_node_searched_with_a_null_window_cannot_store_an_exact_value():
    """The premise of trap (b), pinned rather than argued -- and it was a real hole.

    With `beta == alpha + 1` there is no room between the two bounds, so a scouted node's
    best value lands either at or below `original_alpha` or at or above `beta` and the
    entry it stores is a bound, never a value. Measured over this sample at depth 6, the
    share of stores flagged EXACT falls from **11.6% to 2.2%** (11.6% to 1.5% over the
    reference positions at depth 7).

    This exists because the mutation check found it missing. A mutant that forces the
    EXACT flag on every scouted node changed the stored flags (285 to 368 over thirty
    positions), changed the tree (8 of 30 positions at depth 6, 12 of 30 at depth 7) and
    still produced **0 wrong values and 0 different moves over all 1,050 records** — a
    soundness-shaped fault that nothing in the suite could see. Recording the flag
    composition is what makes it visible.
    """
    shares = {}
    for spec in (NO_SCOUT, SCOUT):
        agent = NegamaxAgent(6, scouting=spec)
        table = type(agent.transposition.start())
        original, counts = table.store, [0, 0]

        def counting(self, key, depth, flag, value, move, _o=original, _c=counts):
            _c[0] += 1
            _c[1] += flag == EXACT
            return _o(self, key, depth, flag, value, move)

        table.store = counting
        try:
            for board in positions(seed=967, count=15):
                agent.choose_move(board)
        finally:
            table.store = original
        shares[spec.name] = counts[1] / counts[0]
    # Pinned tightly on purpose: both numbers are deterministic, and the mutant this test
    # exists for moves the scouted share by only 2.147% -> 2.737%. A loose bound would let
    # it through, which is exactly what the first version of this test did.
    assert shares["no_scout"] > 0.08, shares
    assert shares["negascout"] < 0.025, shares


def test_the_table_stops_answering_and_starts_ordering():
    """Trap (b): a node searched with a null window can never store an exact value --
    with `beta == alpha + 1` there is no room between the two bounds -- so its entry is a
    bound, and a later full-window search that hits it more often cannot settle its own
    window with it. Over the reference positions the share of stored entries flagged
    EXACT falls from 11.6% to 1.5% at depth 7.

    What is observable from a `SearchStats` alone is the consequence: an entry that can no
    longer *answer* a node still supplies a move to try first, so the share of cutoffs
    caused by the transposition move rises. Phase 8 measured it at 0.2-0.6% in a direct
    search and called it worth nothing; it is an order of magnitude larger here.
    """
    shares = {}
    for spec in (NO_SCOUT, SCOUT):
        cutoffs = by_table = 0
        for board in positions(seed=953, count=15):
            _, stats = NegamaxAgent(6, scouting=spec).choose_move(board)
            cutoffs += stats.cutoffs
            by_table += stats.tt_move_cutoffs
        shares[spec.name] = by_table / cutoffs
    assert shares["no_scout"] < 0.02
    assert shares["negascout"] > shares["no_scout"] * 3, shares


def test_it_is_deterministic():
    board = Bitboard.from_move_string("4455")
    first = NegamaxAgent(7).choose_move(board)[1]
    second = NegamaxAgent(7).choose_move(board)[1]
    assert (first.nodes, first.researches, first.value, first.move) == (
        second.nodes, second.researches, second.value, second.move
    )


def test_both_board_implementations_scout_to_the_same_answer():
    from core.board_naive import NaiveBoard

    for moves in ("", "4455", "443322", "1234567"):
        values = {
            cls.__name__: NegamaxAgent(4, evaluation=ZERO if moves == "" else DEFAULT)
            .choose_move(cls.from_move_string(moves))[1]
            .value
            for cls in (Bitboard, NaiveBoard)
        }
        assert len(set(values.values())) == 1, (moves, values)
