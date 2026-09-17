"""NegaScout: proving a move is *worse* instead of finding out how good it is.

Deck slides 146-149, where it is called principal variation search. It is the last of the
alpha-beta family in this project and the smallest of them: four lines inside the move
loop of `agents/alphabeta.py`.

**The idea.** Alpha-beta asks every move the same question -- "what are you worth?" --
and pays for a full window to answer it. But after Phase 7 the move ordering is good: over
the 200 reference positions the very first move tried causes the cutoff **82-89%** of the
time. So at almost every node the first move is already the best one, and every later move
only has to be shown to be *no better*. That is a much cheaper question. Searching a move
with the window `(alpha, alpha + 1)` -- a window no integer value can sit strictly inside
-- forces one of exactly two outcomes:

* the search fails low, `value <= alpha`: the move is not better than what we already
  have, which is the whole of what we wanted to know, and the narrow window will have
  produced cutoffs deeper in the subtree that a wide window would not have;
* the search fails high, `value > alpha`: the move *might* be better, and we know nothing
  else, so the move has to be searched again with the real window.

**The bet, and the number that decides it.** Every fail-high costs a re-search, so this is
profitable exactly when the ordering is good enough to make them rare -- which is what
slide 148 says in so many words: performance depends heavily on move ordering, and there
is 10-30% to be had when the best successor is searched first. Phases 7, 8 and 9 spent
three phases making the first move the best one; this phase is what converts that into
nodes. The re-search rate is reported in `SearchStats` for exactly that reason: it is not
a diagnostic, it is the quantity the algorithm is betting on.

**The first move at every node is searched with the full window.** Not an optimisation to
be skipped: with nothing believed yet there is no `alpha` worth scouting against, and the
node's exact value has to come from somewhere. Getting that wrong changes answers rather
than speed.

--------------------------------------------------------------------------------------
**The re-search window, which is the classic bug of this algorithm.** After a fail-high we
know the move is worth more than `alpha` and we have a number, `value`, in hand. Two
choices:

* re-search with `(alpha, beta)` -- forget what the scout found. Obviously safe.
* re-search with `(value, beta)` -- keep it. This is the textbook form, and it is safe
  *because this project is fail-soft* (Phase 6 chose that deliberately; Phase 8 measured
  it is worth 1.8% of nodes once there is a table). A fail-soft search that fails high
  returns the best value it actually found rather than the edge of its window, and that
  value is a **lower bound** on the move's true value. Re-searching with a floor that the
  true value is already known to be at or above cannot lose it: if the re-search comes
  back at or below that floor it has, by the same fail-soft guarantee, proved the true
  value is at or below it too, so the two bounds meet and the number is exact.

`narrow` is the switch between them, so the claim above is measured rather than argued.
Under fail-*hard* the first option would be the only correct one, which is the concrete
payoff of a convention Phase 6 adopted and Phase 6 could not yet measure.

--------------------------------------------------------------------------------------
**Why "off" is not a second code path** (design principle 2, the same argument as Phase
6's `pruning=False` and Phase 7's `Ordering(off)`). The window handed to a child is one
expression: `alpha + 1` when this move is being scouted, `beta` otherwise. With scouting
off it is `beta` at every move, which is the Phase 9 search character for character, and a
re-search is impossible because the condition that triggers it tests the same expression.
The two modes reproduce each other's node counts exactly rather than approximately.

**It needs pruning, and does nothing without it.** With `pruning=False` the window is one
no value can escape and `alpha` is never raised, so `alpha + 1` would be a window below
the whole score scale -- meaningless. The agent therefore ignores this setting entirely
when it is not pruning, which makes `can_scout` behave like `can_order` (greyed out when
pruning is off) rather than like `can_transpose`. It also has a useful consequence for the
gate: `--verify --agent negamax` compares the **node count** leaf for leaf and is pinned
to `pruning=False`, so it is a direct test that this phase cannot disturb the baseline.

**What it cannot do is change the move that gets played**, which is what puts it in the
category of Phases 7 and 8 rather than Phase 9, and is why it takes the default. At the
root the first move is searched with the full window and fixes `alpha` at its exact value;
every later move is scouted against that `alpha`, and a scout that fails low returns at
most `alpha`, while the root replaces its best move only on a **strict** improvement. So a
move that is merely equal can never displace the one already held -- the same argument
that has held since Phase 6 -- and a move that really is better is re-searched with a real
window before it is allowed to. The one caveat is not this phase's: under iterative
deepening with `share_order` the killer and history tables built by one iteration order
the next one, and scouting changes which cutoffs happen, so it can reach the root that
way. Phase 9 measured the same effect for the ordering and the table and attributed it to
iterative deepening, which is where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scout:
    """Whether to scout later moves with a null window, and how to re-search.

    The fifth frozen spec, after `Evaluation`, `Ordering`, `Transposition` and
    `Deepening`, and read the same way: a value the agent is configured with, whose
    `name` is carried in every row of every results file.
    """

    enabled: bool = True
    narrow: bool = True  # re-search from the bound the scout found, not from alpha
    min_depth: int = 1  # shallowest remaining depth at which a null window is used

    def __post_init__(self) -> None:
        if self.min_depth < 1:
            raise ValueError(f"min_depth must be at least 1, got {self.min_depth}.")

    @property
    def name(self) -> str:
        """The fifth column thesis point P10 requires beside a node count."""
        if not self.enabled:
            return "no_scout"
        parts = ["negascout"]
        if not self.narrow:
            parts.append("wide")
        if self.min_depth > 1:
            parts.append(f"from{self.min_depth}")
        return "+".join(parts)

    def scouts(self, depth: int) -> bool:
        """Whether a node with this much depth left may scout its later moves.

        Depth 1 is where the switch earns its keep: the children of a depth-1 node are
        horizon nodes that return a static evaluation whatever window they are given, so
        a null window there can save nothing at all and every re-search is pure loss. That
        is an argument; `min_depth` is what turns it into a measurement.
        """
        return self.enabled and depth >= self.min_depth


# Phase 9's search: every move gets the full window.
NO_SCOUT = Scout(enabled=False)

# The default. Null-window scouting from depth 1 up, re-searching from the bound found.
SCOUT = Scout()

# The control for the re-search window: throw the scout's bound away and re-search from
# alpha. Must return the same values; the difference is nodes and nothing else.
WIDE = Scout(narrow=False)

# Leave the degenerate depths alone. A depth-1 or depth-2 node has little or nothing for a
# narrower window to cut, so this is where the phase's own overhead is isolated.
DEEP = Scout(min_depth=3)
