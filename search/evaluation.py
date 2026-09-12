"""How good is a position that nobody has won yet.

A depth-limited search has to stop somewhere, and wherever it stops it needs a number.
This is that number: a static estimate, used at cutoff nodes only, never at a terminal
one -- a finished game has a real value and is never guessed at.

Three components, each switchable on its own, because BUILD_PLAN design principle 2 says
an optimisation whose contribution cannot be measured does not belong in this project.
`Evaluation(threats=False, center=False, parity=False)` is the honest zero baseline: a
minimax that uses it can still force wins inside its horizon, but has no opinion about
anything beyond it.

  * threats  -- of the 69 four-in-a-row windows on the board, the ones holding discs of
                only one colour are the ones that can still be completed. Three discs in
                such a window is worth a lot, two a little, and a window holding both
                colours is dead and worth nothing to anyone.
  * center   -- a disc in column 3 belongs to more of those windows than a disc on the
                edge (Phase 4 counted it: 51 of the 69 lines pass through column 3,
                15 through each edge column), so central discs are weighted higher.
  * parity   -- the genuinely Connect Four-specific one, and the one that is absent from
                the course material. Because columns fill from the bottom and the players
                strictly alternate, a threat is worth more to one player than to the
                other depending on the row it sits on: the first player benefits from
                threats on odd rows and the second from threats on even rows, counting
                rows 1-6 from the bottom. Rows here are 0-based, so "odd row" is row 0,
                2 or 4. This is the rule of thumb from Connect Four theory (Allis 1988),
                applied flatly to every three-disc window rather than only to the ones
                that survive into the endgame -- a deliberate simplification, and its
                quality is something this project measures rather than assumes.

**The hard constraint.** Every value this module returns must be strictly inside the
terminal band of `Score`, otherwise a search could prefer a position that merely looks
good to one that is actually won. The bound is not hoped for, it is computed from the
weights below and checked at import time.

Positions are read through the `Board` protocol, so the same evaluation runs on the
naive board and the bitboard alike. It packs what it reads into two 42-bit integers of
its own, which is what makes the 69 window tests two masked ands instead of eight
lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.rules import CELLS, COLS, CONNECT, DIRECTIONS, ROWS, Board
from core.types import EMPTY, Player, Score

# Weights. Module constants rather than fields, so that no caller can push a score out of
# the range the search depends on. Deliberately not tuned to death: BUILD_PLAN names this
# the one component of the project with no objectively correct answer, and timeboxes it.
THREE = 18  # a window with three of one colour and room for the fourth
TWO = 3  # a window with two
CENTER = 2  # per disc, multiplied by the centrality of its column
PARITY = 6  # a three-window whose gap sits on the right row for its owner

# (0, 1, 2, 3, 2, 1, 0): distance from the centre column, inverted.
_CENTRALITY: tuple[int, ...] = tuple(COLS // 2 - abs(col - COLS // 2) for col in range(COLS))

# Local bit layout, unrelated to the bitboard's: 6 bits per column, no sentinel, because
# nothing here ever shifts. bit index = col * ROWS + row.
_CELL_AT: tuple[tuple[int, int], ...] = tuple(
    (row, col) for col in range(COLS) for row in range(ROWS)
)


def _windows() -> tuple[int, ...]:
    """Every set of four cells in a line, as a bitmask. There are 69 of them."""
    found = []
    for col in range(COLS):
        for row in range(ROWS):
            for d_row, d_col in DIRECTIONS:
                cells = [(row + step * d_row, col + step * d_col) for step in range(CONNECT)]
                if all(0 <= r < ROWS and 0 <= c < COLS for r, c in cells):
                    found.append(sum(1 << (c * ROWS + r) for r, c in cells))
    return tuple(found)


WINDOWS = _windows()

# Loose but real upper bound: every window at its most valuable, plus every disc of one
# player in the centre column. Unreachable in practice -- it does not need to be tight,
# it needs to be below MIN_WIN.
MAX_MAGNITUDE = len(WINDOWS) * (max(THREE, TWO) + PARITY) + (CELLS // 2) * max(_CENTRALITY) * CENTER

if MAX_MAGNITUDE >= Score.MIN_WIN:
    raise ValueError(
        f"Evaluation can reach {MAX_MAGNITUDE}, which is not strictly inside the terminal "
        f"band (a win is worth at least {Score.MIN_WIN}). A search using these weights "
        "could prefer a heuristic estimate over a real win. Lower the weights."
    )

# Row parity that favours each player, 0-based from the bottom. Row 0 is the 1st row in
# the usual 1-based statement of the rule, so "odd rows favour the first player" is
# row % 2 == 0 here.
_FAVOURED_PARITY = {Player.ONE: 0, Player.TWO: 1}


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A static estimate of a non-terminal position, with each component switchable."""

    threats: bool = True
    center: bool = True
    parity: bool = True

    @property
    def name(self) -> str:
        on = [
            label
            for label, enabled in (
                ("threats", self.threats),
                ("center", self.center),
                ("parity", self.parity),
            )
            if enabled
        ]
        if not on:
            return "zero"
        return "full" if len(on) == 3 else "+".join(on)

    def __call__(self, board: Board, perspective: Player) -> int:
        """Score `board` from `perspective`'s side. Positive means that player is better.

        Defined for non-terminal positions. A finished game has a real value on the
        `Score` scale and the search never asks this function about one.
        """
        if not (self.threats or self.center or self.parity):
            return 0

        one = two = 0
        center_one = center_two = 0
        for index, (row, col) in enumerate(_CELL_AT):
            occupant = board.cell(row, col)
            if occupant == EMPTY:
                continue
            if occupant == Player.ONE:
                one |= 1 << index
                center_one += _CENTRALITY[col]
            else:
                two |= 1 << index
                center_two += _CENTRALITY[col]

        score = CENTER * (center_one - center_two) if self.center else 0

        if self.threats or self.parity:
            for window in WINDOWS:
                mine = one & window
                theirs = two & window
                if mine:
                    if theirs:
                        continue  # both colours present: nobody can ever complete it
                    discs, sign, owner = mine, 1, Player.ONE
                elif theirs:
                    discs, sign, owner = theirs, -1, Player.TWO
                else:
                    continue

                count = discs.bit_count()
                if self.threats:
                    if count == 3:
                        score += sign * THREE
                    elif count == 2:
                        score += sign * TWO
                if self.parity and count == 3:
                    # Exactly one bit is left once three discs are removed: the gap.
                    gap = (window ^ discs).bit_length() - 1
                    if gap % ROWS % 2 == _FAVOURED_PARITY[owner]:
                        score += sign * PARITY

        return score if perspective is Player.ONE else -score


DEFAULT = Evaluation()
ZERO = Evaluation(threats=False, center=False, parity=False)
