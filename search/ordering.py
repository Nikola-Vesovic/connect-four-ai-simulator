"""The order in which a search tries columns.

Alpha-beta cuts a node off as soon as it finds a move the opponent will never allow. How
much of the tree it therefore never looks at depends almost entirely on **how early that
move is tried**: with a perfect ordering the cut always comes on the first move and the
leaf count falls from b^d to b^(d/2), which is the whole claim of deck slides 51-56. Slide
148 states the practical form of it -- performance depends heavily on move ordering, with
10-30% to be had from searching the best successor first -- and warns that a random order
gives the opposite.

Phase 6 measured exactly where a fixed centre-first order lands: an effective branching
factor of 3.17-3.83 against a floor of sqrt(7) = 2.65. Inside the predicted band, not at
the good end. This module is what closes part of that gap, and the reason it can is that a
static order is a guess made before the search starts, while the search itself keeps
producing evidence about which moves are good. Two classical ways to use it:

**Killer moves.** When a move causes a beta cutoff at some ply, remember it. Sibling
subtrees at the same ply are near-identical positions -- they differ by one move made
somewhere above -- so a move that refuted one of them very often refutes the next. Two
slots per ply, most recent first; a third slot was measured and bought nothing.

**The history heuristic.** A cutoff also says something position-independent: this column
has been refuting things. A table counts it, weighted by `depth * depth`, because a
refutation found high in the tree stood for a much larger subtree than one found near the
leaves. Indexed by the side to move as well as by the column, since a column that is good
for one player is not automatically good for the other.

Both are **only ever a reordering**. Neither can change the value a search returns -- that
is BUILD_PLAN design principle 3, and here it is structural rather than something to
verify: this module never sees a score, only a list of legal columns. The worst a bug in
it can do is make the search slow.

**The slot Phase 8 fills.** The strongest ordering source of all is the transposition
table's stored best move, which is a real answer for this exact position rather than a
guess carried over from a neighbouring one, so it goes in front of the killers. It is
passed in rather than looked up here, because this module still must not be able to see a
score; all it is told is "try this column first". Whether one arrives is the transposition
switch's business, which keeps the two phases separately attributable -- and the search
counts cutoffs caused by a table move apart from cutoffs caused by a killer, so the
question of which source is doing the work is answered by measurement.

**Why the "off" setting is not a second code path** (design principle 2, as with pruning
in Phase 6): `Ordering(killers=False, history=False)` runs the same function over empty
killer slots and an all-zero history table. Nothing gets promoted, the sort is stable and
therefore a no-op, and what comes out is the base order -- so the static configuration
reproduces Phase 6's node counts exactly rather than merely approximately, and the two
cannot drift apart.

**One consequence worth stating because a test pins it.** At the root the tables are still
empty -- the killer slots, the history counts, and from Phase 8 the transposition table
too, which is why the root deliberately never probes it. So the root always tries columns
in the base order whatever the flags say. The move the search finally plays is therefore
unchanged by any of this, and every playing-strength number measured in Phases 5 and 6
carries over untouched. Ordering changes the cost of the answer, not the answer -- and not
even which of several equally good moves gets named.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.rules import COLS, Board
from core.types import Move

# Centre outwards: 3, 2, 4, 1, 5, 0, 6. Ties (columns 2 and 4 are equally central) go to
# the left, arbitrarily but consistently.
#
# The centre is worth preferring because of how the 69 four-in-a-row lines are distributed:
# 51 of them pass through column 3 and only 15 through each edge column, and the two cells
# that belong to the most lines (13 each) are both in column 3. Counted from the board
# geometry, not quoted.
CENTER_FIRST: tuple[Move, ...] = tuple(
    sorted(range(COLS), key=lambda col: (abs(col - COLS // 2), col))
)

# The order a program gets for free if nobody thinks about it. Kept because slide 148's
# warning is worth measuring rather than repeating, and because it is the honest zero
# point for "what is ordering worth" -- centre-first is already an optimisation.
LEFT_TO_RIGHT: tuple[Move, ...] = tuple(range(COLS))

BASE_ORDERS: dict[str, tuple[Move, ...]] = {
    "center_first": CENTER_FIRST,
    "left_to_right": LEFT_TO_RIGHT,
    # Reshuffled at every node; the tuple is only the list the shuffle starts from.
    "random": CENTER_FIRST,
}

# Two killer slots per ply, measured: one is slightly worse, three is better at some
# depths and worse at others and costs more bookkeeping either way.
KILLER_SLOTS = 2


@dataclass(frozen=True, slots=True)
class Ordering:
    """How a search should order moves, as a value. Frozen, like `Evaluation`."""

    base: str = "center_first"
    killers: bool = True
    history: bool = True

    def __post_init__(self) -> None:
        if self.base not in BASE_ORDERS:
            raise ValueError(
                f"Unknown base order {self.base!r}; expected one of {sorted(BASE_ORDERS)}."
            )

    @property
    def name(self) -> str:
        """Carried in every results row, because a node count without it means nothing."""
        parts = [self.base]
        if self.killers:
            parts.append("killers")
        if self.history:
            parts.append("history")
        return "+".join(parts)

    @property
    def is_random(self) -> bool:
        return self.base == "random"

    def start(self, max_ply: int, seed: int | None = None) -> MoveOrder:
        """The working state for one search. Never shared between searches."""
        return MoveOrder(self, max_ply, seed)


class MoveOrder:
    """What one search has learned about which moves to try first.

    Deliberately per-search and thrown away afterwards. Carrying killers and history from
    one move to the next would make a search depend on which positions happened to be
    searched before it, and with that goes the property that a given position always
    produces the same tree -- which every measurement in this thesis relies on. Phase 9
    revisits it, because iterative deepening is the one place where handing the tables
    from one iteration to the next is both standard and harmless.
    """

    __slots__ = ("base", "rng", "use_killers", "use_history", "killers", "history",
                 "promoted", "tt_promoted")

    def __init__(self, spec: Ordering, max_ply: int, seed: int | None = None) -> None:
        self.base = BASE_ORDERS[spec.base]
        self.rng = random.Random(seed) if spec.is_random else None
        self.use_killers = spec.killers
        self.use_history = spec.history
        self.killers: list[list[Move]] = [[-1] * KILLER_SLOTS for _ in range(max_ply + 2)]
        self.history: tuple[list[int], list[int]] = ([0] * COLS, [0] * COLS)
        # How many of the columns just returned were moved to the front, and how many of
        # those came from the transposition table rather than the killer slots. The search
        # reads both immediately, before recursing can overwrite them, and uses them to
        # credit a cutoff to the source that actually caused it.
        self.promoted = 0
        self.tt_promoted = 0

    def columns(self, board: Board, ply: int, tt_move: Move = -1) -> list[Move]:
        """The legal columns at this node, best guess first.

        `tt_move` is the column the transposition table remembers for this exact
        position, or -1. It is tried first when there is one: a killer is a refutation
        borrowed from a sibling, while this one was computed here.
        """
        ordered = [col for col in self.base if board.can_play(col)]
        if self.rng is not None:
            self.rng.shuffle(ordered)

        front = []
        if tt_move >= 0 and board.can_play(tt_move):
            front.append(tt_move)
        tt_promoted = len(front)

        if self.use_killers:
            for killer in self.killers[ply]:
                if killer >= 0 and killer not in front and board.can_play(killer):
                    front.append(killer)

        promoted = len(front)
        if front:
            ordered = front + [col for col in ordered if col not in front]

        if self.use_history:
            table = self.history[board.player_to_move - 1]
            tail = ordered[promoted:]
            # Stable, so columns nothing is known about keep the base order among them.
            tail.sort(key=lambda col: -table[col])
            ordered[promoted:] = tail

        self.promoted = promoted
        self.tt_promoted = tt_promoted
        return ordered

    def cutoff(self, board: Board, col: Move, ply: int, depth: int) -> None:
        """Record that `col` refuted this node.

        Called once the move has been undone, so `board.player_to_move` is the side that
        owns the node rather than the side that answered it.
        """
        if self.use_killers:
            slots = self.killers[ply]
            if slots[0] != col:
                slots[1] = slots[0]
                slots[0] = col
        if self.use_history:
            self.history[board.player_to_move - 1][col] += depth * depth


# Phase 6's ordering, kept as a named value so that "with the ordering work switched off"
# is a setting rather than an old commit.
STATIC = Ordering(killers=False, history=False)

# The default from Phase 7 onwards.
DYNAMIC = Ordering()

# The two failure modes slide 148 warns about. For measuring, not for playing.
LEFT = Ordering(base="left_to_right", killers=False, history=False)
RANDOM = Ordering(base="random", killers=False, history=False)
