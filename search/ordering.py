"""The order in which a search tries columns.

Only the static order for now. Alpha-beta's pruning power depends almost entirely on
trying good moves first, so Phase 7 adds transposition-table moves, killers and the
history heuristic here. The greedy agent breaks its ties by the same order, so "most
central" means one thing across the whole project.
"""

from __future__ import annotations

from core.rules import COLS
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
