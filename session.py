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
takes over two minutes, which in a window is indistinguishable from a crash. The tables
`UNPRUNED_WORST_SECONDS` and `PRUNED_WORST_SECONDS` hold the *measured* worst case of a
single move over sixteen real positions, and the cap is simply the length of the table:
**the interface offers exactly the depths that have been measured, and no others.** That
way the cap cannot quietly drift away from the evidence for it. The numbers are also shown
next to the slider, so the cost of a setting is visible before it is paid.
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
from core.rules import Board, winning_line
from core.types import GameResult, Move, Player
from search.evaluation import DEFAULT, ZERO, Evaluation
from search.stats import UNLIMITED, Budget, SearchStats

# Worst case of ONE move in seconds, depth 1 upwards, measured 2026-09-15 over sixteen
# positions (empty board plus random 2/4/6/8-ply openings) with the full evaluation, on
# the development machine. The empty board alone would be misleading in both directions:
# it is the *worst* case for an unpruned search, which has the most legal moves there, and
# the *best* case for alpha-beta, whose centre-first ordering is close to ideal on a
# symmetric position. Depth 8 pruned costs 2.3s here against 0.13s on the empty board.
UNPRUNED_WORST_SECONDS: tuple[float, ...] = (0.001, 0.001, 0.009, 0.064, 0.442, 3.13, 20.9)
PRUNED_WORST_SECONDS: tuple[float, ...] = (
    0.001, 0.001, 0.004, 0.011, 0.057, 0.150, 1.36, 2.32, 30.9,
)

MAX_DEPTH_UNPRUNED = len(UNPRUNED_WORST_SECONDS)
MAX_DEPTH_PRUNED = len(PRUNED_WORST_SECONDS)


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
            spec.depth, pruning=spec.pruning, evaluation=spec.evaluation, seed=seed
        ),
        uses_depth=True,
        uses_pruning=True,
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
    def max_depth(self) -> int:
        return MAX_DEPTH_PRUNED if self.prunes else MAX_DEPTH_UNPRUNED

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
        """Measured worst case for one move at this setting, or None if it does not search."""
        if not self.spec.uses_depth:
            return 0.0 if self.controller is Controller.MACHINE else None
        table = PRUNED_WORST_SECONDS if self.prunes else UNPRUNED_WORST_SECONDS
        return table[min(self.depth, len(table)) - 1]

    def normalised(self) -> SeatSpec:
        """The same spec with the depth inside the range that has been measured.

        Turning pruning off lowers the cap from 9 to 7, so a depth that was fine a moment
        ago may not be; clamping here means no caller can end up with a setting whose cost
        is unknown.
        """
        depth = max(1, min(self.depth, self.max_depth))
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

    def plan(self, budget: Budget = UNLIMITED) -> tuple[Move, SearchStats]:
        """Ask the agent in the seat to move for its move. Reads; changes nothing.

        Safe to call from another thread for exactly that reason: `Agent.choose_move`
        already hands the search a private copy of the board, so the game state this
        object holds is only ever touched by whoever calls `commit`.
        """
        if self.is_over:
            raise ValueError("The game is over; there is no move to plan.")
        agent = self.agents[self.seat_to_move]
        if agent is None:
            raise ValueError("The seat to move is a human; there is nothing to ask.")
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
