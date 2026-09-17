"""One game of Connect Four, with a configurable player in each seat. No pygame here.

This is the half of Phase 3 that is *what should happen*: whose turn it is, which agent
sits in which seat, whether the position is finished, whether a click is a legal move,
what the last search cost. The other half -- *what is drawn* -- is `gui/` and `game.py`.
The split is not tidiness. It is what lets the whole of the game flow be tested without a
window, and it is why `experiments/` never has to import pygame to stay authoritative.

Three things live here that the window needs and the experiment harness does not:

* **`ALGORITHMS`** -- the picker. One entry per algorithm the user can choose, carrying
  the label, the constructor, and which of the switches apply to it. Adding NegaScout in
  Phase 10 and MCTS in Phase 11 is one line each. It is deliberately *not* the same table
  as `experiments.match.AGENTS`: that one maps a command-line word to a ready-made agent,
  this one carries what a user interface needs to know, and merging them would put
  presentation concerns inside the measurement harness.

* **`SeatSpec`** -- what the user chose for one seat, as a value. Every optimisation is a
  field on it, which is BUILD_PLAN design principle 2 ("every optimisation must be
  switchable") in the form the mentor asked for. It is frozen, so changing a setting
  produces a new spec and cannot half-apply.

* **the measured cost tables** -- see below. They bound the depth slider.

**Why the depth is capped, and how the cap was chosen.** A depth-8 plain minimax move
takes over two minutes, which in a window is indistinguishable from a crash. The four
tables below hold the *measured* worst case of a single move, one table per optimisation
level, and each cap is simply the length of its table: **the interface offers exactly the
depths that have been measured, and no others.** That way the cap cannot quietly drift
away from the evidence for it. The numbers are also shown next to the slider, so the cost
of a setting is visible before it is paid.

One consequence is the clearest thing the panel says about this project: **each
optimisation lengthens its own table.** Pruning takes the reachable depth from 7 to 9,
move ordering from 9 to 11, and the transposition table from 11 to 13, and those are
measurements rather than claims.

The tables form a *ladder*, and a configuration that skips a rung is costed by the last
rung it completed -- a search with a table but no move ordering is priced as if it had
neither. That is deliberately conservative rather than precise: the alternative is one
table per combination, which is six sweeps to measure and keep honest, and the direction
of the approximation is checked rather than assumed (`tests/test_session.py`).

**Phase 9 ends the ladder rather than extending it.** Every rung above exists because a
fixed-depth search has no answer until it finishes, so the only way to bound a move is to
bound the depth and measure what that costs. An iterative search can be stopped, so the
bound becomes the budget and the depth stops being a setting at all: the fourth checkbox
greys the depth control out, releases the cap, and puts the depth *reached* in the panel
as a result. It is also the only switch in the window that makes the worst case smaller.

**Phase 10 does not extend the ladder either, and for the opposite reason to Phase 9's.**
NegaScout only ever makes a search cheaper at the depths the slider offers, so a
configuration with it switched on is priced by the rung it shares with the configuration
without it. That keeps the tables at four and the promise the panel makes conservative --
the quoted worst case is one the search can no longer actually incur -- which is the
direction the ladder has always had to approximate in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from agents.alphabeta import NegamaxAgent
from agents.base import Agent
from agents.greedy_agent import GreedyAgent
from agents.minimax import MinimaxAgent
from agents.random_agent import RandomAgent
from core.bitboard import Bitboard
from core.rules import CELLS, Board, winning_line
from core.types import GameResult, Move, Player
from search.deepening import DIRECT, ITERATIVE, Deepening
from search.evaluation import DEFAULT, ZERO, Evaluation
from search.ordering import DYNAMIC, STATIC, Ordering
from search.scouting import NO_SCOUT, SCOUT, Scout
from search.stats import UNLIMITED, Budget, SearchStats
from search.transposition import NO_TABLE, TABLE, Transposition

# Worst case of ONE move in seconds, depth 1 upwards, re-measured 2026-09-15 over a stated
# sample -- the empty board plus four random openings at each of 2, 4, 6, 8, 10 and 12
# plies, seed 2026 -- with the full evaluation, on the development machine.
#
# The sample matters and Phase 7 found out why. Phase 3 measured sixteen positions it did
# not record and got 30.9s for a pruned depth-9 move; twenty-five positions find 45.5s.
# **A worst case estimated over too few positions is an under-estimate**, which is the
# same lesson as thesis point P11 at a different scale, so the sample is written down here
# rather than described.
#
# The empty board is included deliberately and is never the worst case except for the
# unpruned search: it is the *best* case for alpha-beta, whose centre-first ordering is
# close to ideal on a symmetric position, and the worst for a search with no cutoffs,
# which has the most legal moves there.
UNPRUNED_WORST_SECONDS: tuple[float, ...] = (0.001, 0.001, 0.009, 0.064, 0.442, 3.13, 20.9)
PRUNED_WORST_SECONDS: tuple[float, ...] = (
    0.001, 0.001, 0.003, 0.011, 0.068, 0.222, 1.754, 3.934, 45.527,
)
ORDERED_WORST_SECONDS: tuple[float, ...] = (
    0.001, 0.001, 0.003, 0.008, 0.029, 0.082, 0.222, 0.598, 2.533, 6.346, 18.665,
)
# Phase 8, the same twenty-five positions: alpha-beta with move ordering AND the table.
# Two entries longer than Phase 7's, so the checkbox is worth two more plies of reachable
# depth -- and the entry that says what the phase is worth where a user can feel it is
# depth 11, which cost 18.7 s in Phase 7 and costs 6.7 s now.
TABLE_WORST_SECONDS: tuple[float, ...] = (
    0.001, 0.001, 0.003, 0.007, 0.025, 0.060, 0.167, 0.343, 1.224, 2.570, 6.652,
    10.739, 34.252,
)

MAX_DEPTH_UNPRUNED = len(UNPRUNED_WORST_SECONDS)
MAX_DEPTH_PRUNED = len(PRUNED_WORST_SECONDS)
MAX_DEPTH_ORDERED = len(ORDERED_WORST_SECONDS)
MAX_DEPTH_TABLE = len(TABLE_WORST_SECONDS)

# Phase 9. With iterative deepening and a clock there is **no cost table at all**, because
# there is nothing left to bound: the search stops when the time is up, so the worst case
# of a move is the budget the user chose and not a property of the position. The cap goes
# to the only ceiling the game itself imposes -- a search cannot usefully look further
# ahead than there are empty cells -- and the interface is honest about the change by
# greying the depth out and showing the depth actually *reached* in the panel instead.
#
# This is the one number in the project that a checkbox makes *smaller*: the worst move
# the window permits drops from 34.3 s (depth 13 with the table) to whatever is picked here.
MAX_DEPTH_DEEPENING = CELLS
TIME_BUDGETS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


class Controller(str, Enum):
    """Who supplies the moves for a seat."""

    HUMAN = "human"
    MACHINE = "machine"


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """One entry in the agent picker."""

    key: str
    label: str
    factory: Callable[["SeatSpec", int | None], Agent]
    uses_depth: bool = False
    uses_pruning: bool = False
    uses_ordering: bool = False
    uses_transposition: bool = False
    uses_deepening: bool = False
    uses_scouting: bool = False
    uses_evaluation: bool = False


# The picker. Order is the order it cycles in: cheapest and simplest first.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "random": AlgorithmSpec("random", "Random", lambda spec, seed: RandomAgent(seed=seed)),
    "greedy": AlgorithmSpec("greedy", "Greedy", lambda spec, seed: GreedyAgent(seed=seed)),
    "minimax": AlgorithmSpec(
        "minimax",
        "Minimax",
        lambda spec, seed: MinimaxAgent(spec.depth, evaluation=spec.evaluation, seed=seed),
        uses_depth=True,
        uses_evaluation=True,
    ),
    "negamax": AlgorithmSpec(
        "negamax",
        "Negamax",
        lambda spec, seed: NegamaxAgent(
            spec.search_depth,
            pruning=spec.pruning,
            evaluation=spec.evaluation,
            ordering=spec.ordering,
            transposition=spec.transposition_spec,
            deepening=spec.deepening_spec,
            scouting=spec.scouting_spec,
            seed=seed,
        ),
        uses_depth=True,
        uses_pruning=True,
        uses_ordering=True,
        uses_transposition=True,
        uses_deepening=True,
        uses_scouting=True,
        uses_evaluation=True,
    ),
}

ALGORITHM_KEYS: tuple[str, ...] = tuple(ALGORITHMS)


@dataclass(frozen=True, slots=True)
class SeatSpec:
    """Everything the user chose for one seat. Frozen: a change produces a new spec."""

    controller: Controller = Controller.HUMAN
    algorithm: str = "negamax"
    depth: int = 6
    pruning: bool = True
    move_ordering: bool = True
    transposition: bool = True
    deepening: bool = False
    scouting: bool = True
    time_budget: float = 1.0
    threats: bool = True
    center: bool = True
    parity: bool = True

    @property
    def spec(self) -> AlgorithmSpec:
        return ALGORITHMS[self.algorithm]

    @property
    def prunes(self) -> bool:
        """Whether this configuration actually prunes -- the flag only counts if the
        chosen algorithm has somewhere to apply it."""
        return self.spec.uses_pruning and self.pruning

    @property
    def can_order(self) -> bool:
        """Whether the ordering switch means anything here -- regardless of its value.

        It needs pruning as well as an algorithm that prunes: with no cutoffs there is
        nothing for a better move order to cut earlier, and nothing for the killer table
        to learn. Phase 5 proved that direction for minimax and Phase 6 proved the
        opposite for alpha-beta, so the panel can grey the switch out on solid ground.

        Deliberately independent of `move_ordering` itself: a checkbox that switched
        itself off permanently would be a trap.
        """
        return self.spec.uses_ordering and self.prunes

    @property
    def orders(self) -> bool:
        """Whether the learned ordering is actually in effect. This is what the depth cap
        and the cost estimate depend on."""
        return self.can_order and self.move_ordering

    @property
    def ordering(self) -> Ordering:
        """Killer moves and the history heuristic, or Phase 6's plain centre-first order.

        Shared instances, for the same reason the evaluation hands back shared ones: an
        agent configured in the window must report the same name as the same agent
        configured in an experiment.
        """
        return DYNAMIC if self.move_ordering else STATIC

    @property
    def can_transpose(self) -> bool:
        """Whether the transposition switch means anything here -- regardless of its value.

        Unlike `can_order` this does **not** require pruning, and the difference is a
        measurement rather than a matter of taste. Move ordering needs cutoffs to bring
        forward, so with pruning off it has nothing to do. A table needs only that the
        same position turn up twice, which happens in any tree; with pruning off it is in
        fact at its most effective, because the unpruned tree is the one with the most
        repeats in it.
        """
        return self.spec.uses_transposition

    @property
    def transposition_active(self) -> bool:
        """Whether the table is actually in effect. This is what the depth cap and the
        cost estimate depend on -- the same two-property split Phase 7 needed, where
        conflating them was a real bug."""
        return self.can_transpose and self.transposition

    @property
    def transposition_spec(self) -> Transposition:
        return TABLE if self.transposition else NO_TABLE

    @property
    def can_deepen(self) -> bool:
        """Whether the deepening switch means anything here -- regardless of its value.

        Like `can_transpose` and unlike `can_order`, it does **not** require pruning:
        searching 1, 2, 3... plies and answering from the last one that finished is the
        same algorithm with or without cutoffs. What it does require is a search with a
        depth to iterate towards, which greedy and random have not got.
        """
        return self.spec.uses_deepening

    @property
    def deepens(self) -> bool:
        """Whether the clock is actually in charge. This is what the depth cap, the
        estimate and the budget depend on -- the same two-property split Phase 7 needed,
        where conflating the two was a real bug."""
        return self.can_deepen and self.deepening

    @property
    def deepening_spec(self) -> Deepening:
        return ITERATIVE if self.deepening else DIRECT

    @property
    def can_scout(self) -> bool:
        """Whether the NegaScout switch means anything here -- regardless of its value.

        This one sides with `can_order` rather than with `can_transpose`: a null window is
        a *pruning* window, so with pruning off there is nothing for it to do, and the
        agent ignores the setting entirely in that case. Greying it out is therefore an
        honest description of what the search will do and not a matter of taste.
        """
        return self.spec.uses_scouting and self.prunes

    @property
    def scouts(self) -> bool:
        """Whether null-window scouting is actually in effect -- the two-property split
        every optimisation since Phase 7 has needed.

        Unlike its three predecessors this one feeds no depth cap: NegaScout is priced by
        the rung it shares with the configuration without it, because it can only make a
        search cheaper and the ladder must never quote a cost below what can be incurred.
        """
        return self.can_scout and self.scouting

    @property
    def scouting_spec(self) -> Scout:
        return SCOUT if self.scouting else NO_SCOUT

    @property
    def budget(self) -> Budget:
        """What the seat's agent is allowed to spend on one move.

        The first budget in this project that is not `UNLIMITED`, and the reason the
        window's anti-freeze guarantee stops depending on a measured depth cap: a search
        that can be stopped is bounded by the number chosen here rather than by a table.
        """
        return Budget(time=self.time_budget) if self.deepens else UNLIMITED

    @property
    def search_depth(self) -> int:
        """The depth handed to the agent. Under a clock it is a ceiling, not a target."""
        return MAX_DEPTH_DEEPENING if self.deepens else self.depth

    @property
    def max_depth(self) -> int:
        if self.deepens:
            return MAX_DEPTH_DEEPENING
        if not self.prunes:
            return MAX_DEPTH_UNPRUNED
        if not self.orders:
            return MAX_DEPTH_PRUNED
        return MAX_DEPTH_TABLE if self.transposition_active else MAX_DEPTH_ORDERED

    @property
    def evaluation(self) -> Evaluation:
        chosen = Evaluation(threats=self.threats, center=self.center, parity=self.parity)
        # Hand back the shared instances where they match, so an agent configured in the
        # window reports the same name as the same agent configured in an experiment:
        # both agents suffix their name only when the evaluation is not the default, and
        # they test that by identity.
        for known in (DEFAULT, ZERO):
            if chosen == known:
                return known
        return chosen

    @property
    def estimated_seconds(self) -> float | None:
        """Measured worst case for one move at this setting, or None if it does not search.

        With the clock in charge this stops being an estimate at all and becomes the
        budget itself -- which is the strongest form of the guarantee the depth caps were
        invented to provide, and the reason Phase 9 is the phase that *lowers* the worst
        case the interface permits.
        """
        if not self.spec.uses_depth:
            return 0.0 if self.controller is Controller.MACHINE else None
        if self.deepens:
            return self.time_budget
        if not self.prunes:
            table = UNPRUNED_WORST_SECONDS
        elif not self.orders:
            table = PRUNED_WORST_SECONDS
        elif self.transposition_active:
            table = TABLE_WORST_SECONDS
        else:
            table = ORDERED_WORST_SECONDS
        return table[min(self.depth, len(table)) - 1]

    def normalised(self) -> SeatSpec:
        """The same spec with the depth inside the range that has been measured.

        Turning an optimisation off lowers the cap -- 13 with the table, 11 with ordering,
        9 without it, 7 with no pruning at all -- so a depth that was fine a moment ago
        may not be; clamping here means no caller can end up with a setting whose cost is
        unknown.

        Ticking *iterative deepening* does the opposite and raises the depth to the
        released cap, rather than leaving the search silently limited to whatever number
        the slider happened to be showing. The promise the checkbox makes is "as deep as
        fits in the time", and a stale 6 would quietly break it. The depth control greys
        out at the same moment, so nothing the user can still change is being overridden.
        """
        wanted = self.max_depth if self.deepens else self.depth
        depth = max(1, min(wanted, self.max_depth))
        return self if depth == self.depth else replace(self, depth=depth)

    def build(self, seed: int | None = None) -> Agent | None:
        """The agent this seat plays with, or None for a human."""
        if self.controller is Controller.HUMAN:
            return None
        settled = self.normalised()
        return settled.spec.factory(settled, seed)


HUMAN_SEAT = SeatSpec(controller=Controller.HUMAN)
MACHINE_SEAT = SeatSpec(controller=Controller.MACHINE)

# The three modes the mentor asked for, as presets. The underlying model is per seat, so
# these are a convenience and not a separate concept -- every combination is reachable by
# setting the two seats independently.
MODES: tuple[tuple[str, tuple[Controller, Controller]], ...] = (
    ("Čovek vs mašina", (Controller.HUMAN, Controller.MACHINE)),
    ("Mašina vs mašina", (Controller.MACHINE, Controller.MACHINE)),
    ("Čovek vs čovek", (Controller.HUMAN, Controller.HUMAN)),
)


@dataclass(frozen=True, slots=True)
class MoveRecord:
    seat: int
    player: Player
    column: Move
    stats: SearchStats | None  # None when a human played it


@dataclass(slots=True)
class SeatTotals:
    """Everything one seat has spent this game."""

    moves: int = 0
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0
    seconds: float = 0.0

    def add(self, stats: SearchStats | None) -> None:
        self.moves += 1
        if stats is None:
            return
        self.nodes += stats.nodes
        self.leaves += stats.leaves
        self.cutoffs += stats.cutoffs
        self.seconds += stats.elapsed

    @property
    def nodes_per_move(self) -> float:
        return self.nodes / self.moves if self.moves else 0.0


class Session:
    """One game in progress, plus the configuration of both seats.

    The caller drives it. `drop` is a human move, `plan` asks the agent for one and
    `commit` applies it. `plan` is split from `commit` on purpose: it only *reads*, so the
    window can run it on a worker thread and stay responsive, and only the main thread
    ever changes the game.
    """

    def __init__(
        self,
        seats: tuple[SeatSpec, SeatSpec] = (HUMAN_SEAT, MACHINE_SEAT),
        *,
        board_class=Bitboard,
        seed: int = 2026,
    ) -> None:
        self.board_class = board_class
        self.seed = seed
        self.seats: list[SeatSpec] = [seats[0].normalised(), seats[1].normalised()]
        self.agents: list[Agent | None] = [None, None]
        self.board: Board = board_class()
        self.history: list[MoveRecord] = []
        self.totals: list[SeatTotals] = [SeatTotals(), SeatTotals()]
        self.games_played = 0
        self.new_game()

    # ------------------------------------------------------------------ configuration

    def _seed_for(self, index: int) -> int:
        """A per-game, per-seat seed. Only the random agent uses it today; MCTS will in
        Phase 11. The experiment harness has the rigorous version of this."""
        return (self.seed * 1_000_003 + self.games_played * 31 + index) % (2**31)

    def _rebuild(self, index: int) -> None:
        self.agents[index] = self.seats[index].build(self._seed_for(index))

    def set_seat(self, index: int, spec: SeatSpec) -> SeatSpec:
        """Apply a new configuration to one seat, from the next move onwards.

        Returns what was actually stored, which may differ from what was asked for: the
        depth is clamped to the range that has been measured.
        """
        self.seats[index] = spec.normalised()
        self._rebuild(index)
        return self.seats[index]

    def set_mode(self, controllers: tuple[Controller, Controller]) -> None:
        for index, controller in enumerate(controllers):
            self.set_seat(index, replace(self.seats[index], controller=controller))

    def new_game(self) -> None:
        """Fresh board, same configuration."""
        self.games_played += 1
        self.board = self.board_class()
        self.history = []
        self.totals = [SeatTotals(), SeatTotals()]
        for index in range(2):
            self._rebuild(index)
            agent = self.agents[index]
            if agent is not None:
                agent.new_game(self._seed_for(index))

    # ------------------------------------------------------------------------- state

    @property
    def seat_to_move(self) -> int:
        """0 or 1. Seat 0 always starts, so it is Player.ONE."""
        return self.board.moves_played % 2

    @property
    def spec_to_move(self) -> SeatSpec:
        return self.seats[self.seat_to_move]

    @property
    def is_over(self) -> bool:
        return self.board.is_terminal()

    @property
    def result(self) -> GameResult:
        return self.board.result()

    @property
    def awaiting_human(self) -> bool:
        return not self.is_over and self.spec_to_move.controller is Controller.HUMAN

    @property
    def awaiting_machine(self) -> bool:
        return not self.is_over and self.spec_to_move.controller is Controller.MACHINE

    @property
    def mode_label(self) -> str:
        for label, controllers in MODES:
            if tuple(seat.controller for seat in self.seats) == controllers:
                return label
        return "Mašina vs čovek"

    def last_stats(self, index: int) -> SearchStats | None:
        for record in reversed(self.history):
            if record.seat == index:
                return record.stats
        return None

    def last_move(self) -> MoveRecord | None:
        return self.history[-1] if self.history else None

    def winning_cells(self) -> tuple[tuple[int, int], ...] | None:
        return winning_line(self.board)

    # ------------------------------------------------------------------------ moving

    def drop(self, col: Move) -> bool:
        """A human plays a column. False -- and nothing changes -- if that is not allowed.

        Every reason to refuse is checked here rather than in the window: the game is
        over, it is not a human's turn, or the column is full. The interface only has to
        turn a click into a number.
        """
        if not self.awaiting_human or col not in self.board.legal_moves():
            return False
        self.commit(col, None)
        return True

    def plan(self, budget: Budget | None = None) -> tuple[Move, SearchStats]:
        """Ask the agent in the seat to move for its move. Reads; changes nothing.

        Safe to call from another thread for exactly that reason: `Agent.choose_move`
        already hands the search a private copy of the board, so the game state this
        object holds is only ever touched by whoever calls `commit`.

        `None` means "whatever this seat is allowed", which is `UNLIMITED` for everything
        before Phase 9 and a clock once iterative deepening is switched on. An explicit
        budget overrides it, which is what the experiment harness does -- and why nothing
        in `experiments/` can accidentally inherit a wall-clock limit from the interface.
        """
        if self.is_over:
            raise ValueError("The game is over; there is no move to plan.")
        agent = self.agents[self.seat_to_move]
        if agent is None:
            raise ValueError("The seat to move is a human; there is nothing to ask.")
        if budget is None:
            budget = self.spec_to_move.budget
        return agent.choose_move(self.board, budget)

    def commit(self, col: Move, stats: SearchStats | None) -> None:
        """Put the move on the board and record what it cost."""
        if self.is_over:
            raise ValueError("The game is over; no further move can be played.")
        seat = self.seat_to_move
        player = self.board.player_to_move
        self.board.play(col)
        self.history.append(MoveRecord(seat, player, col, stats))
        self.totals[seat].add(stats)
