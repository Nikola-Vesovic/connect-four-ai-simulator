"""Phase 3: the game flow, tested without a window.

The reason this file can exist at all is the split the phase is built on: *what should
happen* lives in `session.py` and *what is drawn* lives in `gui/`. Everything below is the
first half -- turns, legality, terminal positions, mode switching, and whether the
switches in the panel reach the agent they claim to configure.

Two of these tests are worth more than the rest and are worth naming:

* `test_the_pruning_switch_changes_the_search_and_not_the_answer` is the mentor's request
  turned into an assertion. The checkbox is only meaningful if flipping it changes the
  number of positions examined by orders of magnitude *and* leaves the chosen move alone.
* `test_nothing_headless_imports_pygame` guards BUILD_PLAN 2.3. If the measurement harness
  ever picks up a pygame import it can no longer be run under PyPy, and that is the
  project's escape hatch for speed.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.board_naive import NaiveBoard
from core.rules import COLS, ROWS, winning_line
from core.types import EMPTY, GameResult, Player
from session import (
    ALGORITHM_KEYS,
    ALGORITHMS,
    MAX_DEPTH_DEEPENING,
    MAX_DEPTH_ORDERED,
    MAX_DEPTH_PRUNED,
    MAX_DEPTH_TABLE,
    MAX_DEPTH_UNPRUNED,
    MCTS_WORST_SECONDS_PER_SIMULATION,
    MODES,
    ORDERED_WORST_SECONDS,
    PRUNED_WORST_SECONDS,
    SIMULATION_BUDGETS,
    TABLE_WORST_SECONDS,
    TIME_BUDGETS,
    UNPRUNED_WORST_SECONDS,
    Controller,
    SeatSpec,
    Session,
)
from agents.mcts import MctsAgent
from search.deepening import DIRECT, ITERATIVE
from search.playout import SQRT2
from search.scouting import NO_SCOUT, SCOUT
from search.stats import UNLIMITED, Budget
from search.transposition import NO_TABLE, TABLE

HUMAN = SeatSpec(controller=Controller.HUMAN)
RANDOM = SeatSpec(controller=Controller.MACHINE, algorithm="random")
GREEDY = SeatSpec(controller=Controller.MACHINE, algorithm="greedy")


def machine(algorithm: str = "negamax", **changes) -> SeatSpec:
    return replace(
        SeatSpec(controller=Controller.MACHINE, algorithm=algorithm, depth=4), **changes
    )


def play_out(session: Session, limit: int = 42) -> None:
    """Run a machine-versus-machine game to its end, the way the window would."""
    for _ in range(limit):
        if session.is_over:
            return
        move, stats = session.plan()
        session.commit(move, stats)


# --------------------------------------------------------------------------- the split


def test_nothing_headless_imports_pygame():
    """`session` and the whole of `experiments/` must stay runnable without a display.

    Checked in a fresh interpreter, because inside the test suite pygame may already have
    been imported by the smoke test and `sys.modules` would no longer prove anything.

    Phase 12 makes this assertion do a second job. macOS starts worker processes with
    `spawn`, so a worker **is** a fresh interpreter that re-imports whichever module
    holds the work function -- `experiments.tournament` -- and, as `__mp_main__`, the
    module that was run. Asserting that importing them leaves pygame out is therefore
    exactly the assertion that a tournament worker is pygame-free. Phase 13 adds
    `experiments.benchmark`, whose workers are spawned the same way.

    Phase 14 adds **matplotlib** to the same assertion, and `experiments.ablation`, whose
    control sweep spawns workers too. A drawing library has no business in a measurement
    process or in the application, and `experiments.plots` is included deliberately: it
    draws every figure in the thesis and still must not import matplotlib until a drawing
    function is actually called, which is what keeps the data shaping testable -- and this
    suite green -- on a machine where matplotlib is not installed at all.
    """
    root = Path(__file__).resolve().parent.parent
    program = (
        "import sys;"
        "import session, experiments.match, experiments.scaling, experiments.reference,"
        " experiments.tournament, experiments.runner, experiments.benchmark,"
        " experiments.ablation, experiments.plots, agents.oracle;"
        "assert 'pygame' not in sys.modules, sorted(m for m in sys.modules if 'pygame' in m);"
        "assert 'matplotlib' not in sys.modules,"
        " sorted(m for m in sys.modules if 'matplotlib' in m);"
        "print('clean')"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "clean" in done.stdout


# ------------------------------------------------------------------------ configuration


def test_every_algorithm_in_the_picker_can_actually_play():
    """The picker is a promise: whatever it offers has to produce a legal move."""
    for key in ALGORITHM_KEYS:
        session = Session((machine(key), machine(key)))
        move, stats = session.plan()
        assert move in session.board.legal_moves(), key
        assert stats.move == move


def test_the_depth_offered_is_exactly_the_depth_that_was_measured():
    """The cap and the cost table are the same object, so they cannot drift apart.

    The ordering of the four caps is itself a result: each optimisation lengthens its own
    table, so pruning is worth two plies of reachable depth, move ordering another two,
    and the transposition table one more.
    """
    assert MAX_DEPTH_UNPRUNED == len(UNPRUNED_WORST_SECONDS)
    assert MAX_DEPTH_PRUNED == len(PRUNED_WORST_SECONDS)
    assert MAX_DEPTH_ORDERED == len(ORDERED_WORST_SECONDS)
    assert MAX_DEPTH_TABLE == len(TABLE_WORST_SECONDS)
    assert MAX_DEPTH_TABLE >= MAX_DEPTH_ORDERED > MAX_DEPTH_PRUNED > MAX_DEPTH_UNPRUNED
    for spec, table in (
        (machine(pruning=True, move_ordering=True, transposition=True), TABLE_WORST_SECONDS),
        (machine(pruning=True, move_ordering=True, transposition=False), ORDERED_WORST_SECONDS),
        (machine(pruning=True, move_ordering=False), PRUNED_WORST_SECONDS),
        (machine(pruning=False), UNPRUNED_WORST_SECONDS),
    ):
        for depth in range(1, spec.max_depth + 1):
            assert replace(spec, depth=depth).estimated_seconds == table[depth - 1]


def test_the_cost_ladder_never_over_promises():
    """The four tables are a ladder, and a configuration that skips a rung is costed by
    the last rung it completed -- a table without move ordering is priced as if it had
    neither. That is an approximation, and the direction of it is the thing that matters:
    it must never quote a cost lower than the configuration can actually incur.

    Checked here rather than assumed, because the alternative -- one measured table per
    combination of switches -- is six sweeps to keep honest instead of four.
    """
    for depth in range(1, MAX_DEPTH_PRUNED + 1):
        quoted = machine(depth=depth, move_ordering=False, transposition=True)
        fallback = machine(depth=depth, move_ordering=False, transposition=False)
        assert quoted.estimated_seconds == fallback.estimated_seconds
        assert quoted.max_depth == fallback.max_depth


def test_the_estimate_falls_when_an_optimisation_is_switched_on():
    """What the panel promises the user: the number next to the slider is the cost of the
    settings actually chosen, so ticking a box visibly buys something."""
    for depth in range(1, MAX_DEPTH_PRUNED + 1):
        ordered = machine(depth=depth, pruning=True, move_ordering=True).estimated_seconds
        plain = machine(depth=depth, pruning=True, move_ordering=False).estimated_seconds
        assert ordered <= plain, depth
    for depth in range(1, MAX_DEPTH_ORDERED + 1):
        with_table = machine(depth=depth, transposition=True).estimated_seconds
        without = machine(depth=depth, transposition=False).estimated_seconds
        assert with_table <= without, depth


def test_depth_is_clamped_to_what_has_been_measured():
    session = Session((HUMAN, machine(depth=99)))
    stored = session.set_seat(1, machine(depth=99))
    assert stored.depth == MAX_DEPTH_TABLE

    # Turning an optimisation off lowers the ceiling, so a depth that was legal a moment
    # ago has to come down with it -- three times now, once per switch.
    stored = session.set_seat(1, replace(stored, transposition=False))
    assert stored.depth == MAX_DEPTH_ORDERED
    stored = session.set_seat(1, replace(stored, move_ordering=False))
    assert stored.depth == MAX_DEPTH_PRUNED
    stored = session.set_seat(1, replace(stored, pruning=False))
    assert stored.depth == MAX_DEPTH_UNPRUNED
    assert session.set_seat(1, machine(depth=0)).depth == 1


def test_switches_only_apply_where_they_mean_something():
    """Greedy has no depth and no evaluation; the panel greys them out on this basis."""
    assert not ALGORITHMS["greedy"].uses_depth
    assert not ALGORITHMS["greedy"].uses_evaluation
    assert not ALGORITHMS["minimax"].uses_pruning  # Phase 5's search has nothing to prune
    assert not ALGORITHMS["minimax"].uses_ordering  # and so nothing for ordering to steer
    assert ALGORITHMS["negamax"].uses_pruning
    assert ALGORITHMS["negamax"].uses_ordering
    assert ALGORITHMS["negamax"].uses_transposition

    # Ordering needs cutoffs to bring forward, so it is dead while pruning is off.
    assert not machine("negamax", pruning=False, move_ordering=True).orders
    assert not machine("negamax", pruning=False, move_ordering=True).can_order
    assert machine("negamax", pruning=True, move_ordering=True).orders
    # But the switch stays clickable when it is off, or it could never be turned back on.
    assert machine("negamax", pruning=True, move_ordering=False).can_order
    assert not machine("negamax", pruning=True, move_ordering=False).orders

    # The table is the one optimisation that does NOT need pruning, because it needs only
    # that a position turn up twice. With pruning off it is at its most effective.
    assert machine("negamax", pruning=False, transposition=True).can_transpose
    assert machine("negamax", pruning=False, transposition=True).transposition_active
    assert machine("negamax", pruning=True, transposition=False).can_transpose
    assert not machine("negamax", pruning=True, transposition=False).transposition_active
    assert not machine("minimax").can_transpose


# ---------------------------------------------------------------------------- Phase 11


def test_mcts_is_a_fifth_entry_in_the_picker_and_not_a_sixth_checkbox():
    """The rule Phase 3 and Phase 10 both applied, coming out the other way for once.

    Alpha-beta and NegaScout are checkboxes because they provably cannot change the move
    played -- they buy the same answer for less work. MCTS is an algorithm by exactly that
    test: it computes no minimax value, never calls the evaluation function, and plays
    different moves on purpose. So the five alpha-beta switches mean nothing for it and
    the panel greys every one of them out.
    """
    assert ALGORITHM_KEYS == ("random", "greedy", "minimax", "negamax", "mcts")
    spec = machine("mcts")
    assert spec.searches_a_tree
    assert not any(
        (spec.spec.uses_pruning, spec.spec.uses_ordering, spec.spec.uses_transposition,
         spec.spec.uses_scouting, spec.spec.uses_evaluation, spec.spec.uses_depth)
    )
    assert not spec.prunes and not spec.can_order and not spec.can_transpose
    assert not spec.can_scout
    # The one switch it shares, and it means the same thing in both: stop on the clock
    # instead of at a fixed amount of work.
    assert spec.can_deepen
    assert not machine("negamax").searches_a_tree


def test_the_mcts_switches_reach_the_agent_and_are_named_there():
    spec = machine("mcts", simulations=250)
    assert spec.build(seed=1).name == "mcts-250"
    assert isinstance(spec.build(seed=1), MctsAgent)
    assert replace(spec, greedy_rollouts=False).build().name == "mcts-250-uct_random"
    assert replace(spec, tree_reuse=True).build().name == "mcts-250-uct_greedy+reuse"
    assert replace(spec, solver=True).build().name == "mcts-250-uct_greedy+solver"
    assert replace(spec, exploration=0.7).build().name == "mcts-250-uct_greedy+c0.7"
    assert spec.playout_spec.exploration == SQRT2


def test_the_mcts_estimate_is_linear_in_the_simulations_it_was_measured_over():
    """The cost model is different in kind from the four depth tables above, and the
    reason is a property of the algorithm: MCTS runs a fixed number of simulations
    whatever the position, so one worst case per simulation covers every position. A tree
    search is the opposite -- thesis point P11 is the story of that difference -- which is
    why the depth caps needed a table each."""
    for budget in SIMULATION_BUDGETS:
        spec = machine("mcts", simulations=budget)
        assert spec.estimated_seconds == pytest.approx(
            budget * MCTS_WORST_SECONDS_PER_SIMULATION["greedy"]
        )
        cheap = replace(spec, greedy_rollouts=False)
        assert cheap.estimated_seconds < spec.estimated_seconds
    # Ticking the clock replaces the estimate with the budget, exactly as it does for
    # alpha-beta -- the one switch in the window that makes the worst case smaller.
    timed = machine("mcts", simulations=25_000, deepening=True, time_budget=0.5)
    assert timed.estimated_seconds == 0.5
    assert timed.budget == Budget(time=0.5)
    assert timed.build().iterations > max(SIMULATION_BUDGETS)


def test_the_simulation_budget_does_not_disturb_the_depth_setting():
    """MCTS has no depth to clamp, so ticking its clock must not write 42 into a field the
    other four algorithms share."""
    session = Session((HUMAN, machine("negamax", depth=6)))
    session.set_seat(1, machine("mcts", depth=6, deepening=True))
    assert session.seats[1].depth == 6
    stored = session.set_seat(1, machine("negamax", depth=6, deepening=True))
    assert stored.depth == MAX_DEPTH_DEEPENING

    # A pruning flag set on an algorithm that cannot prune must not raise the depth cap.
    assert not machine("minimax", pruning=True).prunes
    assert machine("minimax", pruning=True).max_depth == MAX_DEPTH_UNPRUNED


def test_the_transposition_switch_reaches_the_agent_and_is_named_there():
    assert machine(transposition=True).build().transposition is TABLE
    assert machine(transposition=False).build().transposition is NO_TABLE
    assert machine(transposition=False).build().name.endswith("-no_table")
    assert "no_table" not in machine(transposition=True).build().name


def test_the_deepening_switch_reaches_the_agent_and_is_named_there():
    assert machine(deepening=True).build().deepening is ITERATIVE
    assert machine(deepening=False).build().deepening is DIRECT
    assert machine(deepening=True).build().name.endswith("-iterative+pv+order")
    assert "iterative" not in machine(deepening=False).build().name

    # Like the table and unlike the ordering, it does not need pruning to mean something.
    assert machine("negamax", pruning=False, deepening=True).can_deepen
    assert machine("negamax", pruning=False, deepening=True).deepens
    assert machine("negamax", deepening=False).can_deepen  # still clickable when off
    assert not machine("negamax", deepening=False).deepens
    assert not machine("minimax").can_deepen
    assert not machine("greedy").can_deepen


def test_the_scouting_switch_reaches_the_agent_and_is_named_there():
    """Phase 10's checkbox. It sides with the ordering rather than with the table: a null
    window is a pruning window, so with alpha-beta off there is nothing for it to do and
    the agent ignores it -- which is what the panel says by greying it out.

    It is also the first switch since Phase 8 that does not touch the depth cap. NegaScout
    can only make a search cheaper, so it is priced by the rung it shares with the search
    without it, and the ladder stays conservative.
    """
    assert ALGORITHMS["negamax"].uses_scouting
    assert not ALGORITHMS["minimax"].uses_scouting
    assert machine(scouting=True).build().scouting is SCOUT
    assert machine(scouting=False).build().scouting is NO_SCOUT
    assert machine(scouting=False).build().name.endswith("-no_scout")
    assert "scout" not in machine(scouting=True).build().name

    assert not machine("negamax", pruning=False, scouting=True).can_scout
    assert not machine("negamax", pruning=False, scouting=True).scouts
    assert machine("negamax", pruning=True, scouting=False).can_scout  # still clickable
    assert not machine("negamax", pruning=True, scouting=False).scouts
    assert not machine("minimax").can_scout

    for depth in range(1, MAX_DEPTH_TABLE + 1):
        with_scout = machine(depth=depth, scouting=True)
        without = machine(depth=depth, scouting=False)
        assert with_scout.max_depth == without.max_depth
        assert with_scout.estimated_seconds == without.estimated_seconds


def test_the_clock_replaces_the_depth_cap_rather_than_extending_it():
    """The point of the phase, stated as a property of the interface.

    Every other switch lengthens its own measured cost table, so the worst move the
    window permits gets *bigger*. This one hands the bound to the budget: the depth is
    released to the only ceiling the game itself has, and the quoted cost becomes the
    number the user chose -- which is smaller than the deepest fixed-depth setting.
    """
    deep = machine(deepening=True, time_budget=1.0)
    assert deep.max_depth == MAX_DEPTH_DEEPENING > MAX_DEPTH_TABLE
    assert deep.estimated_seconds == 1.0
    assert deep.estimated_seconds < TABLE_WORST_SECONDS[-1]
    assert deep.budget == Budget(time=1.0)
    assert machine(deepening=False).budget is UNLIMITED
    for budget in TIME_BUDGETS:
        assert replace(deep, time_budget=budget).estimated_seconds == budget


def test_ticking_the_clock_raises_the_depth_and_unticking_puts_it_back():
    """A stale depth would quietly break the promise the checkbox makes -- "as deep as
    fits in the time" is not "as deep as the slider happened to be showing"."""
    session = Session((HUMAN, machine(depth=6)))
    stored = session.set_seat(1, machine(depth=6, deepening=True))
    assert stored.depth == MAX_DEPTH_DEEPENING
    stored = session.set_seat(1, replace(stored, deepening=False))
    assert stored.depth == MAX_DEPTH_TABLE


def test_a_seat_under_a_clock_is_asked_for_a_move_with_that_clock():
    """`Session.plan()` with no argument has to hand the agent the seat's own budget, or
    the switch would be visible in the panel and absent from the search."""
    session = Session((machine(deepening=True, time_budget=0.1), HUMAN))
    _, stats = session.plan()
    assert stats.iterations > 1
    assert stats.elapsed < 0.5
    assert stats.depth_cap == MAX_DEPTH_DEEPENING
    # An explicit budget still wins, which is what keeps `experiments/` authoritative.
    _, capped = session.plan(Budget(depth=3))
    assert capped.depth == 3


def test_the_evaluation_switches_reach_the_agent():
    for changes, expected in (
        ({}, "full"),
        ({"threats": False, "center": False, "parity": False}, "zero"),
        ({"center": False}, "threats+parity"),
    ):
        agent = machine(**changes).build()
        assert agent.evaluation.name == expected
        assert agent.evaluation == machine(**changes).evaluation


def test_a_human_seat_has_no_agent():
    session = Session((HUMAN, RANDOM))
    assert HUMAN.build() is None
    assert session.agents[0] is None
    assert session.agents[1] is not None


# ------------------------------------------------------------------------------- turns


def test_seat_zero_moves_first_and_the_turn_alternates():
    session = Session((HUMAN, HUMAN))
    assert session.seat_to_move == 0
    assert session.board.player_to_move is Player.ONE
    session.drop(3)
    assert session.seat_to_move == 1
    assert session.board.player_to_move is Player.TWO
    session.drop(3)
    assert session.seat_to_move == 0


def test_a_human_move_is_refused_when_it_is_not_allowed():
    session = Session((HUMAN, RANDOM))

    assert session.drop(3) is True
    assert session.board.moves_played == 1

    # Seat 1 is a machine: a click must not move for it.
    assert session.drop(4) is False
    assert session.board.moves_played == 1


def test_a_full_column_is_refused():
    session = Session((HUMAN, HUMAN))
    for _ in range(ROWS):
        assert session.drop(0) is True
    assert not session.board.can_play(0)
    assert not session.is_over  # one column cannot hold four of the same colour

    before = session.board.moves_played
    assert session.drop(0) is False
    assert session.board.moves_played == before


def test_a_finished_game_accepts_nothing_further():
    session = Session((HUMAN, HUMAN))
    for col in (0, 1, 0, 1, 0, 1, 0):  # seat 0 wins vertically in column 0
        session.drop(col)

    assert session.is_over
    assert session.result is GameResult.ONE_WINS
    assert session.awaiting_human is False
    assert session.drop(5) is False
    with pytest.raises(ValueError):
        session.plan()
    with pytest.raises(ValueError):
        session.commit(5, None)


def test_the_winning_four_are_found_in_every_direction():
    for moves, expected in (
        ((0, 1, 0, 1, 0, 1, 0), ((0, 0), (1, 0), (2, 0), (3, 0))),  # vertical
        ((0, 0, 1, 1, 2, 2, 3), ((0, 0), (0, 1), (0, 2), (0, 3))),  # horizontal
        ((0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3), ((0, 0), (1, 1), (2, 2), (3, 3))),  # diagonal
    ):
        board = NaiveBoard.from_moves(moves)
        assert board.winner() is Player.ONE
        assert set(winning_line(board)) == set(expected)

    assert winning_line(NaiveBoard.from_moves((0, 1, 0, 1))) is None


def test_plan_reads_the_position_and_changes_nothing():
    """The property the worker thread depends on: planning a move is not playing it."""
    session = Session((machine("minimax", depth=4), HUMAN))
    before = [
        [session.board.cell(row, col) for col in range(COLS)] for row in range(ROWS)
    ]
    moves_before = session.board.moves_played

    move, stats = session.plan()

    assert session.board.moves_played == moves_before
    assert [[session.board.cell(r, c) for c in range(COLS)] for r in range(ROWS)] == before
    assert all(cell == EMPTY for row in before for cell in row)

    session.commit(move, stats)
    assert session.board.moves_played == moves_before + 1


# ----------------------------------------------------------------------------- the toggles


def test_the_pruning_switch_changes_the_search_and_not_the_answer():
    """The mentor's request as an assertion, and the demonstration for the defence.

    Same algorithm, same depth, same evaluation, same position. Only the checkbox moves.
    Alpha-beta must visit dramatically fewer positions and must still choose the same
    column -- BUILD_PLAN design principle 3.
    """
    opening = (3, 3, 4, 4)
    counts = {}
    for pruning in (True, False):
        session = Session((machine(depth=6, pruning=pruning), HUMAN))
        for col in opening:
            session.commit(col, None)
        move, stats = session.plan()
        counts[pruning] = (move, stats.nodes, stats.value, stats.cutoffs)

    pruned, plain = counts[True], counts[False]
    assert pruned[0] == plain[0]  # same move
    assert pruned[2] == plain[2]  # same value
    assert pruned[1] * 10 < plain[1]  # at least an order of magnitude fewer nodes
    assert pruned[3] > 0 and plain[3] == 0  # and cutoffs are what did it


def test_the_ordering_switch_changes_the_search_and_not_the_answer():
    """Phase 7's checkbox, held to exactly the same standard as Phase 6's.

    The bar is lower than pruning's on purpose: pruning removes orders of magnitude,
    ordering removes a factor. What must be identical is more than pruning guarantees --
    not only the value but the column, because the killer and history tables are still
    empty when the root picks its order.
    """
    opening = (3, 3, 4, 4, 2, 5)
    counts = {}
    for ordering in (True, False):
        session = Session((machine(depth=6, move_ordering=ordering), HUMAN))
        for col in opening:
            session.commit(col, None)
        move, stats = session.plan()
        counts[ordering] = (move, stats.value, stats.nodes, stats.first_cutoff_rate)

    ordered, plain = counts[True], counts[False]
    assert ordered[0] == plain[0]  # same move
    assert ordered[1] == plain[1]  # same value
    assert ordered[2] < plain[2]  # fewer positions examined
    assert ordered[3] > plain[3]  # and the ordering quality is what improved


def test_the_ordering_switch_reaches_the_agent_and_is_named_there():
    assert machine(move_ordering=True).build().ordering.killers
    assert not machine(move_ordering=False).build().ordering.killers
    assert machine(move_ordering=False).build().name.endswith("center_first")
    assert not machine(move_ordering=True).build().name.endswith("center_first")


def test_switching_the_evaluation_off_changes_what_alpha_beta_examines():
    """Phase 6's rule, visible in the panel: a pruned search's node count depends on the
    evaluation, because what can be cut depends on what has been found so far."""
    nodes = {}
    for name, changes in (("full", {}), ("zero", dict(threats=False, center=False, parity=False))):
        session = Session((machine(depth=5, **changes), HUMAN))
        for col in (3, 3, 4, 2):  # an even number of plies, so seat 0 is the one to move
            session.commit(col, None)
        nodes[name] = session.plan()[1].nodes
    assert nodes["full"] != nodes["zero"]


# ------------------------------------------------------------------------------- modes


def test_the_three_modes_set_both_seats():
    session = Session((HUMAN, HUMAN))
    for label, controllers in MODES:
        session.set_mode(controllers)
        assert tuple(seat.controller for seat in session.seats) == controllers
        assert session.mode_label == label
    assert session.mode_label == "Čovek vs čovek"

    session.set_mode((Controller.MACHINE, Controller.HUMAN))
    assert session.mode_label == "Mašina vs čovek"


def test_a_seat_can_be_reconfigured_in_the_middle_of_a_game():
    session = Session((HUMAN, RANDOM))
    session.drop(3)
    session.commit(*session.plan())
    position = [[session.board.cell(r, c) for c in range(COLS)] for r in range(ROWS)]

    session.set_seat(1, GREEDY)

    assert [[session.board.cell(r, c) for c in range(COLS)] for r in range(ROWS)] == position
    assert session.board.moves_played == 2
    assert session.agents[1].name == "greedy"
    assert len(session.history) == 2


def test_machine_versus_machine_plays_a_whole_game_by_itself():
    session = Session((GREEDY, RANDOM))
    play_out(session)
    assert session.is_over
    assert session.result is not GameResult.IN_PROGRESS
    assert session.totals[0].moves + session.totals[1].moves == session.board.moves_played
    assert session.totals[0].nodes > 0  # greedy looks ahead one reply
    assert session.totals[1].nodes == 0  # the random agent does not look at anything


# ------------------------------------------------------------------------- bookkeeping


def test_statistics_accumulate_per_seat_and_a_human_costs_nothing():
    session = Session((HUMAN, GREEDY))
    session.drop(3)
    session.commit(*session.plan())
    session.drop(2)
    session.commit(*session.plan())

    assert session.totals[0].moves == 2
    assert session.totals[0].nodes == 0
    assert session.totals[0].seconds == 0.0
    assert session.last_stats(0) is None

    assert session.totals[1].moves == 2
    assert session.totals[1].nodes > 0
    assert session.totals[1].nodes_per_move == session.totals[1].nodes / 2
    assert session.last_stats(1) is session.history[-1].stats


def test_a_new_game_clears_the_board_but_keeps_the_configuration():
    session = Session((HUMAN, GREEDY))
    session.drop(3)
    session.commit(*session.plan())
    specs = list(session.seats)

    session.new_game()

    assert session.board.moves_played == 0
    assert session.history == []
    assert session.totals[0].moves == session.totals[1].moves == 0
    assert list(session.seats) == specs
    assert session.agents[1] is not None


def test_the_same_seed_replays_the_same_game():
    """Two random agents are the only thing here with any randomness in it. If a rerun
    from the same seed diverges, nothing measured through this object can be trusted."""
    def transcript() -> list[int]:
        session = Session((RANDOM, RANDOM), seed=99)
        play_out(session)
        return [record.column for record in session.history]

    first = transcript()
    assert len(first) > 6
    assert transcript() == first


def test_consecutive_games_in_one_session_are_not_identical():
    """The per-game seed moves on, or machine-versus-machine would replay one game."""
    session = Session((RANDOM, RANDOM), seed=7)
    play_out(session)
    first = [record.column for record in session.history]
    session.new_game()
    play_out(session)
    assert [record.column for record in session.history] != first
