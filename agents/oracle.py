"""The perfect solver: the only agent in this project that is never wrong.

**Benchmarking only.** It is not in the picker, not in `tournament.BUILDERS`, and nothing
can put it in a game by accident: a full-depth search from an early position is hours, and
BUILD_PLAN section 1 reserved this name with "benchmarking only" written next to it.

It exists because of thesis point P2. Greedy scores 98.6% against random, and that number
measures partly how good greedy is and partly how bad random is at punishing -- a win rate
against an opponent is a blunt instrument. The true value of a position is not. This module
is what makes the second kind of measurement possible: it turns a position into the exact
set of moves that preserve its value, so an agent can be scored against the objectively
correct answer rather than against another agent.

Two pieces of arithmetic live here and both are easy to get wrong.

**1. Pons's scale against ours.** The published benchmark scores a position as *22 minus
the number of stones the winner has played when the game ends*, positive when the side to
move wins. Ours is `Score.win_in(p) = WIN - p` with `p` in plies from the root. With `n`
stones on the board and the game ending `p` plies later the winner has played
`ceil((n + p) / 2)`, so

    pons = +-(22 - ceil((n + p) / 2))

`ceil` is not injective, so the conversion back looks ambiguous -- but it is not, because
the parity of `p` is forced: the side to move can only win on an odd ply and only lose on
an even one. That is what makes `from_pons(to_pons(v, n), n) == v` a real round trip and
therefore testable.

**2. A child's value is one ply further from the root than the child thinks it is.**
`stats.value` is always measured in plies from the position that was searched. Negating a
child's value gives the right *sign* and the wrong *distance*, so the negation is followed
by one step away from the root. Getting this wrong changes no outcome and every mate
distance, which is exactly the kind of fault the published scores catch and a win rate
never would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from agents.alphabeta import NegamaxAgent
from agents.base import Agent
from core.rules import CELLS, Board
from core.types import Move, Score
from search.deepening import ITERATIVE
from search.ordering import CENTER_FIRST
from search.stats import UNLIMITED, Budget, SearchStats

# 22 for a 6x7 board: a player places at most CELLS // 2 = 21 stones, so a win on the very
# last one scores 1 rather than 0 and the scale never collides with a draw.
PONS_BASE = CELLS // 2 + 1

# A four-in-a-row needs four stones, so the winner has played at least four.
MAX_PONS = PONS_BASE - 4

WIN, DRAW, LOSS = 1, 0, -1

# Sized by measurement, not chosen: the cost of an exhaustive solve is exponential in the
# EMPTY cells, and `/tmp/p13_wall.py` measured 725 nodes for a 27-stone Middle-Easy
# position against 717,676 for a 16-stone one -- with a tail beyond that which does not
# finish at all. A node budget is the right ceiling rather than a clock because it aborts
# at the same node every time (thesis point P24), so a bounded oracle still reproduces.
DEFAULT_BUDGET = Budget(nodes=1_000_000)


class Unsettled(RuntimeError):
    """The search ran out of budget without proving anything, so there is no ground truth.

    Raised rather than returned, and counted rather than swallowed: a position the oracle
    could not settle is excluded from the sample and the number excluded is reported. A
    benchmark that quietly kept a heuristic guess would be scoring agents against an
    opinion.
    """


def outcome_of(value: int) -> int:
    """The theoretical outcome an exact value stands for: +1 win, 0 draw, -1 loss.

    Only meaningful for an exact value. A heuristic score sits strictly inside the band and
    would be reported as a draw, which is why `Oracle.solve` refuses to return one.
    """
    if value >= Score.MIN_WIN:
        return WIN
    if value <= Score.MAX_LOSS:
        return LOSS
    return DRAW


def one_ply_further(value: int) -> int:
    """Move an exact value one ply away from the root, keeping its band.

    This is the whole of the child-to-parent correction. A draw has no distance to move.
    """
    if value >= Score.MIN_WIN:
        return value - 1
    if value <= Score.MAX_LOSS:
        return value + 1
    return Score.DRAW


def to_pons(value: int, stones: int) -> int:
    """Our root value for a position with `stones` discs on it, on Pons's scale."""
    if value >= Score.MIN_WIN:
        return PONS_BASE - math.ceil((stones + Score.WIN - value) / 2)
    if value <= Score.MAX_LOSS:
        return -(PONS_BASE - math.ceil((stones + value - Score.LOSS) / 2))
    return 0


def from_pons(score: int, stones: int) -> int:
    """The inverse. Well defined because the parity of the distance is forced.

    The side to move places a stone on plies 1, 3, 5..., so if it wins the game ends on an
    odd ply and if it loses on an even one. Of the two distances that satisfy the `ceil`,
    exactly one has the required parity.
    """
    if score == 0:
        return Score.DRAW
    if not -MAX_PONS <= score <= MAX_PONS:
        raise ValueError(f"{score} is outside Pons's scale [-{MAX_PONS}, {MAX_PONS}].")
    stones_played = PONS_BASE - abs(score)
    wants_odd = score > 0
    for plies in (2 * stones_played - 1 - stones, 2 * stones_played - stones):
        if plies >= 1 and (plies % 2 == 1) == wants_odd:
            return Score.win_in(plies) if score > 0 else Score.loss_in(plies)
    raise ValueError(f"no distance of the right parity for score {score} at {stones} stones.")


@dataclass(frozen=True)
class MoveValues:
    """Every legal move of one position, valued exactly, from the mover's own side.

    The published file gives the value of a *position*, not of each move, so it cannot judge
    a move on its own. This is the object that closes that gap, and the three metrics of
    Phase 13 are three different questions asked of it -- which is why they are three
    properties here and not one score.
    """

    values: Mapping[Move, int]
    nodes: int
    seconds: float

    @property
    def value(self) -> int:
        """The position's own value: what the best move preserves."""
        return max(self.values.values())

    @property
    def optimal(self) -> tuple[Move, ...]:
        """Every move that preserves the value -- usually more than one, which is why this
        is never called "the best move"."""
        best = self.value
        return tuple(sorted(m for m, v in self.values.items() if v == best))

    @property
    def discriminating(self) -> bool:
        """False when every legal move preserves the value, i.e. when the position cannot
        tell any two agents apart and measured nothing."""
        return len(self.optimal) < len(self.values)

    @property
    def blunder_available(self) -> bool:
        """False when no legal move can change the theoretical outcome -- which is always
        true in a lost position, where every move loses. A blunder rate of zero over such a
        population is a property of the population and not of the agent."""
        best = outcome_of(self.value)
        return any(outcome_of(v) < best for v in self.values.values())

    def loss(self, move: Move) -> int:
        """How much the move gives away, on the exact scale.

        Inside one outcome class this is exactly the number of extra plies -- a win in 11
        instead of a win in 5 loses 6. Across classes it is about 10,000 (one class) or
        20,000 (two), so a mean over both kinds is dominated by blunders and has to be
        reported decomposed.
        """
        return self.value - self.values[move]

    def is_optimal(self, move: Move) -> bool:
        return self.values[move] == self.value

    def is_blunder(self, move: Move) -> bool:
        """A move that changes the theoretical OUTCOME: win to draw, win to loss, draw to
        loss. A win in 11 played instead of a win in 5 is suboptimal and not a blunder."""
        return outcome_of(self.values[move]) < outcome_of(self.value)


class Oracle(Agent):
    """A bounded full-depth search that returns only values it has proved.

    **It is iterative rather than direct, and that is what makes it bounded.** A fixed-depth
    search has no answer until it finishes, so it cannot be stopped (Phase 9); an iterative
    one always holds the last completed iteration's value. The cap is `depth = CELLS`, and
    any cap at or above the number of empty cells produces the same tree, because every line
    ends at a terminal node before the cap.

    **A proof is much cheaper than an exhaustive solve, and a search can certify its own
    answer without completing.** That is the observation this class rests on and it is worth
    stating precisely, because it is what makes the phase affordable at all:

      * a value in the WIN band at depth d means a forced win exists within d plies, and
        the distance is minimal -- a shorter win would also lie within d and the maximum
        would have preferred it;
      * a value in the LOSS band at depth d means *every* line loses within d, because a
        line surviving past the horizon would return a heuristic score and the maximum at
        the root would be that score instead; and the distance is maximal for the same
        reason;
      * a value of exactly 0 proves nothing on its own -- it is also what a drawn heuristic
        looks like -- so a draw is accepted only when the iteration that produced it reached
        `CELLS - stones`, i.e. when the tree really was exhausted.

    Anything else raises `Unsettled`.

    **The transposition table is deliberately not shared between the children of one
    position**, although siblings do overlap and sharing would be faster. Phase 8 measured
    what a table that outlives its root does: it can answer a node from a *deeper* search
    than the one being run, and design principle 3 forbids changing the answer even by
    improving it (thesis point P21). Ground truth is the last place to relax that.
    """

    name = "oracle"

    def __init__(self, budget: Budget = DEFAULT_BUDGET) -> None:
        super().__init__(None)
        if budget.time is not None:
            raise ValueError(
                "an oracle budgeted by a wall clock would settle a different set of "
                "positions on every run; use nodes (thesis point P24)."
            )
        self.budget = budget
        self._search = NegamaxAgent(CELLS, deepening=ITERATIVE)

    def solve(self, board: Board) -> tuple[int, SearchStats]:
        """The exact value of a non-terminal position, in plies from that position.

        The depth cap is the number of EMPTY cells, not `CELLS`. An iteration past that
        point re-searches an identical tree -- every line already ends at a terminal node --
        so the cap is exact and the saving is large: a 35-stone End-Easy position needs
        seven iterations rather than forty-two, and it measured 71.3 ms against 14.0.
        """
        remaining = CELLS - board.moves_played
        _, stats = self._search.choose_move(
            board, Budget(nodes=self.budget.nodes, depth=remaining)
        )
        value = stats.value
        if value >= Score.MIN_WIN or value <= Score.MAX_LOSS:
            return value, stats
        if value == Score.DRAW and not stats.aborted and stats.depth >= remaining:
            return value, stats
        raise Unsettled(
            f"{board.moves_played} stones, reached depth {stats.depth} of {remaining} "
            f"in {stats.nodes} nodes and the value is still {value}, which is a "
            "heuristic score."
        )

    def move_values(self, board: Board) -> MoveValues:
        """Value every legal move exactly, from the point of view of the player to move.

        A terminal child is handled here rather than passed to the search, because
        `Agent.choose_move` refuses a finished position -- and the two cases are the two
        ways a game can end: the move just played wins, or it fills the last cell.
        """
        values: dict[Move, int] = {}
        nodes = 0
        seconds = 0.0
        for move in board.legal_moves():
            board.play(move)
            if board.is_terminal():
                values[move] = Score.win_in(1) if board.winner() is not None else Score.DRAW
            else:
                child, stats = self.solve(board)
                values[move] = one_ply_further(-child)
                nodes += stats.nodes
                seconds += stats.elapsed
            board.undo()
        if not values:
            raise ValueError("the position has no legal moves, so it is already over.")
        return MoveValues(values, nodes, seconds)

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        """Play a value-preserving move, the most central of them for determinism.

        The budget is ignored on purpose: an oracle that answered a cheaper question would
        not be an oracle. This exists so the benchmark can be checked end to end against an
        agent that must score 100% optimal and 0% blunders by construction.
        """
        table = self.move_values(board)
        stats.nodes = table.nodes
        stats.value = table.value
        optimal = table.optimal
        return min(optimal, key=CENTER_FIRST.index)
