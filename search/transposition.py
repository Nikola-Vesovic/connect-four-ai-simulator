"""The transposition table: remembering positions the search has already solved.

Deck slide 5 names *transpozicije* and says using them speeds the search up. This is what
that sentence means in practice, and it attacks something neither alpha-beta nor move
ordering can touch. Both of those work on **one** tree: alpha-beta declines to look at
parts of it, move ordering arranges it so that more can be declined. Neither notices that
the same position occurs in the tree many times over, reached by different move orders,
and is searched again from scratch every time.

**How much of the tree that is, measured before any of this was written** (full
fixed-depth tree from the empty board, nodes against distinct positions):

    depth 3    399 nodes    294 distinct    1.36x
    depth 5     19,607       5,677         3.45x
    depth 7    960,792      76,958        12.48x
    depth 8  6,634,026     261,233        25.40x

There are no transpositions at all before ply 3, and the reason is worth a sentence
because it is specific to this game: reaching one position by two move orders requires
*each player* to end up on the same cells, and with one disc each the two orders merely
swap the colours. Three discs is the first point at which the same arrangement can be
built two ways.

Those figures are a **ceiling on a minimax-shaped tree**, not a prediction. Alpha-beta
already refuses to visit most of that tree, so the repeats available to be saved in a
*pruned* search are far fewer. What the table actually saves is measured, not inferred.

--------------------------------------------------------------------------------------
**The key.** `position + mask`, straight off the bitboard -- see `core.rules.Board.key`.
The unusual part is that it is **collision-free rather than a hash**: two positions with
the same key are the same position, provably, so the table cannot ever return an answer
belonging to somewhere else. Chess engines use Zobrist hashing and accept a small
probability of exactly that. Here the whole class of bug is absent, which is why a plain
dictionary is a legitimate implementation and not a shortcut.

**What an entry holds.** The value, the remaining depth it was computed at, a flag saying
whether that value is exact or only a bound, and the best move found. The flag is the
part that is easy to get wrong: alpha-beta is given a window, and a node that leaves its
window early has not computed a value, it has computed a *bound*.

    EXACT  the search stayed inside its window: this is the value
    LOWER  the node cut off, so the true value is at least this (fail-high)
    UPPER  no move reached alpha, so the true value is at most this (fail-low)

Fail-soft is what makes the bound worth storing. Phase 6 measured that fail-soft and
fail-hard visit identical trees and return identical values with no table, and predicted
they would become distinguishable here; a fail-hard search would store the window edge
where a fail-soft one stores the tighter value it actually found.

--------------------------------------------------------------------------------------
**Two implementations, because the trade-off is the interesting part.**

`dict` -- exact. Every position stored is retrievable; nothing is ever evicted, so the
table never gives up information it has. It costs whatever CPython charges for a
dictionary of that size, which is not small.

`array` -- fixed size, two 64-bit arrays, 16 bytes an entry, indexed by a mixed key and
holding the full key for comparison. Bounded memory by construction, at the price of
*index collisions*: two different positions wanting the same slot, where one has to be
thrown away. Replacement is depth-preferred. Both the collision rate and what it costs in
extra nodes are reported rather than assumed.

--------------------------------------------------------------------------------------
**Mate scores are the trap.** `Score.win_in(ply) = WIN - ply` measures the distance to
the win from the **root**, not from the node. A value stored raw and read back at a
different ply is then wrong by the difference. So values are converted to node-relative
on the way in and back on the way out, which is what `to_entry`/`from_entry` do.

Whether that conversion is *load-bearing* in this phase is a separate question, and the
answer turned out to be no -- see the Phase 8 notes in BUILD_PLAN. Every move puts exactly
one disc on the board, so a position's key determines how many discs it has, which fixes
the ply it can occur at within one search from one root; and `depth + ply` is constant, so
it fixes the remaining depth too. Same key, same ply, same depth. The conversion is an
identity here and starts to matter the moment a table outlives the root it was built for.
It is written anyway, because that moment is Phase 9.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass

from core.types import Move, Score

EXACT = 0
LOWER = 1  # the true value is >= the stored one (the node cut off)
UPPER = 2  # the true value is <= the stored one (nothing reached alpha)

# An entry is (depth, flag, value, move). The value is node-relative; see to_entry.
Entry = tuple[int, int, int, Move]

# Packing for the fixed-size table: everything an entry holds in one machine word.
_VALUE_BIAS = Score.WIN  # shifts the value into a non-negative range
_DEPTH_SHIFT = 15
_FLAG_SHIFT = 21
_MOVE_SHIFT = 23

# Fibonacci hashing: the key's low bits are column 0 and nothing else, so indexing an
# array with them straight would pile half the table into a few slots.
_MIX = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1


def to_entry(value: int, ply: int) -> int:
    """Root-relative score in, node-relative score out.

    A win `n` plies below this node is worth `WIN - (ply + n)` to the root and `WIN - n`
    here, so storing adds the ply back on; a loss is the mirror image. Anything in the
    heuristic band is a static judgement about the position and carries no distance, so
    it is stored untouched.
    """
    if value >= Score.MIN_WIN:
        return value + ply
    if value <= Score.MAX_LOSS:
        return value - ply
    return value


def from_entry(value: int, ply: int) -> int:
    """The inverse of `to_entry`, applied at the ply the entry is being read at."""
    if value >= Score.MIN_WIN:
        return value - ply
    if value <= Score.MAX_LOSS:
        return value + ply
    return value


class DictTable:
    """The exact table. Nothing stored is ever lost, because the key never collides."""

    kind = "dict"
    __slots__ = ("capacity", "_entries", "probes", "hits", "stores", "rewrites",
                 "collisions", "overflows", "peak")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: dict[int, Entry] = {}
        self.peak = 0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.probes = self.hits = self.stores = 0
        self.rewrites = self.collisions = self.overflows = 0

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> int:
        return len(self._entries)

    @property
    def fill(self) -> float:
        return len(self._entries) / self.capacity

    def probe(self, key: int) -> Entry | None:
        self.probes += 1
        found = self._entries.get(key)
        if found is not None:
            self.hits += 1
        return found

    def store(self, key: int, depth: int, flag: int, value: int, move: Move) -> None:
        self.stores += 1
        entries = self._entries
        if key in entries:
            # Same position, stored again: the first visit left only a bound, so the
            # probe could not answer this window. Never a different position -- the key
            # is exact -- which is why this counter is not a collision rate.
            self.rewrites += 1
        elif len(entries) >= self.capacity:
            # The cap exists so that a deep search cannot quietly eat the machine. Once
            # it is reached the table stops growing rather than evicting: with an exact
            # key there is no cheap way to choose a victim, and a table that keeps what
            # it found first is easier to reason about than one that thrashes.
            self.overflows += 1
            return
        entries[key] = (depth, flag, value, move)
        if len(entries) > self.peak:
            self.peak = len(entries)


class ArrayTable:
    """Fixed size, 16 bytes an entry, depth-preferred replacement.

    Two parallel arrays rather than one array of objects, because the point of this
    implementation is the memory and a Python object per entry would defeat it. Slot zero
    of `_keys` means empty, so keys are stored offset by one -- the empty board's key is
    genuinely 0 and would otherwise be indistinguishable from an unused slot.
    """

    kind = "array"
    __slots__ = ("capacity", "_mask", "_shift", "_keys", "_data", "probes", "hits",
                 "stores", "rewrites", "collisions", "overflows", "peak", "_filled")

    def __init__(self, capacity: int) -> None:
        if capacity & (capacity - 1):
            raise ValueError(f"Capacity must be a power of two, got {capacity}.")
        self.capacity = capacity
        self._mask = capacity - 1
        self._shift = 64 - capacity.bit_length() + 1
        self._keys = array("q", bytes(8 * capacity))
        self._data = array("q", bytes(8 * capacity))
        self._filled = 0
        self.peak = 0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.probes = self.hits = self.stores = 0
        self.rewrites = self.collisions = self.overflows = 0

    def clear(self) -> None:
        zeros = array("q", bytes(8 * self.capacity))
        self._keys = zeros
        self._data = array("q", bytes(8 * self.capacity))
        self._filled = 0

    @property
    def entries(self) -> int:
        return self._filled

    @property
    def fill(self) -> float:
        return self._filled / self.capacity

    def _slot(self, key: int) -> int:
        return ((key * _MIX) & _MASK64) >> self._shift & self._mask

    def probe(self, key: int) -> Entry | None:
        self.probes += 1
        slot = self._slot(key)
        if self._keys[slot] != key + 1:
            return None  # empty, or somebody else's -- either way, nothing to report
        self.hits += 1
        packed = self._data[slot]
        return (
            packed >> _DEPTH_SHIFT & 0x3F,
            packed >> _FLAG_SHIFT & 0x3,
            (packed & 0x7FFF) - _VALUE_BIAS,
            packed >> _MOVE_SHIFT & 0x7,
        )

    def store(self, key: int, depth: int, flag: int, value: int, move: Move) -> None:
        self.stores += 1
        slot = self._slot(key)
        resident = self._keys[slot]
        if resident == key + 1:
            self.rewrites += 1
        elif resident:
            # Somebody else's entry. Depth-preferred: a deep result cost more to compute
            # and stands for a bigger subtree, so it is the one worth keeping. Equal
            # depths go to the newcomer, which is the common case here because within one
            # search a position always carries the same remaining depth.
            if self._data[slot] >> _DEPTH_SHIFT & 0x3F > depth:
                self.overflows += 1
                return
            self.collisions += 1
        else:
            self._filled += 1
            if self._filled > self.peak:
                self.peak = self._filled
        self._keys[slot] = key + 1
        self._data[slot] = (
            (value + _VALUE_BIAS)
            | depth << _DEPTH_SHIFT
            | flag << _FLAG_SHIFT
            | move << _MOVE_SHIFT
        )


TABLES = {"dict": DictTable, "array": ArrayTable}

# 2^20 entries, and the cap is a memory bound rather than a tuning knob. MEASURED by
# filling both tables to capacity: the dictionary costs 141.9 MiB (141.9 bytes an entry,
# which is what CPython charges for a hash table of tuples of boxed integers) and the
# fixed-size table 17.0 MiB (two arrays of 2^20 machine words, 16 bytes an entry by
# construction plus the object headers). The deepest search the window offers holds
# 127,214 entries, 12% of the cap, so in practice neither is anywhere near it -- which is
# exactly what a cap is for.
DEFAULT_CAPACITY = 1 << 20


@dataclass(frozen=True, slots=True)
class Transposition:
    """How a search should use a transposition table, as a value. Frozen, like
    `Evaluation` and `Ordering`, and carried in the name of every results row."""

    enabled: bool = True
    kind: str = "dict"
    capacity: int = DEFAULT_CAPACITY
    persistent: bool = False
    reads: bool = True

    def __post_init__(self) -> None:
        if self.kind not in TABLES:
            raise ValueError(f"Unknown table kind {self.kind!r}; expected one of {sorted(TABLES)}.")
        if self.capacity < 1:
            raise ValueError(f"A table needs room for at least one entry, got {self.capacity}.")

    @property
    def name(self) -> str:
        if not self.enabled:
            return "no_table"
        parts = [self.kind]
        if self.capacity != DEFAULT_CAPACITY:
            # Two tables of different sizes are two different optimisations, and a results
            # row that called them both "array" would be unreadable.
            parts[0] += f"{self.capacity // 1024}k" if self.capacity >= 1024 else str(self.capacity)
        if not self.reads:
            parts.append("write_only")
        if self.persistent:
            parts.append("persistent")
        return "+".join(parts)

    def start(self) -> DictTable | ArrayTable | None:
        """A fresh table for one search, or None if the optimisation is switched off.

        `None` rather than a do-nothing table on purpose. A null object would put a
        method call on the hot path at every node so that the "off" setting could share a
        code path with the "on" one, and the cost of that would show up in exactly the
        measurement this phase exists to make.
        """
        return None if not self.enabled else TABLES[self.kind](self.capacity)


# On from Phase 8 onwards, in the same way DYNAMIC became the default ordering in Phase 7.
TABLE = Transposition()

# Phase 7's search, kept as a named value so that "with the table switched off" is a
# setting rather than an old commit.
NO_TABLE = Transposition(enabled=False)

# The fixed-size alternative, and the controlled experiment that isolates the cost of
# keeping the table from the benefit of reading it: `WRITE_ONLY` probes and stores exactly
# as the real thing does but never acts on what it finds, so it provably searches the same
# tree and every second of difference is bookkeeping.
ARRAY = Transposition(kind="array")
WRITE_ONLY = Transposition(reads=False)

# Survives between moves. Measured, not recommended -- see BUILD_PLAN Phase 8.
PERSISTENT = Transposition(persistent=True)
