"""Iterative deepening: searching 1, 2, 3... plies instead of d plies once.

Deck slide 46 calls it *progresivno produbljivanje*. It looks wasteful and the first
question anybody asks is why re-searching everything from scratch at every depth is not
simply d times the work. Two answers, and only the second one is the point.

The arithmetic answer. A tree with branching factor b has about b^d leaves, so the
searches at depths 1..d-1 together cost about b^(d-1) * b/(b-1) -- the sum of a geometric
series, dominated by its last term. At b = 7 that is a 17% surcharge on the final
iteration. Phase 7 predicted a crossover for move ordering on the same kind of reasoning
and there was none, so the surcharge here is measured rather than quoted: the `NAIVE`
setting below carries nothing from one iteration to the next, so with the transposition
table switched off it costs **exactly** the sum of the separate searches -- an identity,
asserted rather than argued, which turns the textbook estimate into something falsifiable.
Measured, it is 1.33x-1.75x rather than 1.17x, and it **oscillates with the parity of the
target depth** for the reason Phase 6 found: the price of one more ply is itself
alternating (about x2 going to an even depth and about x6 going to an odd one), so the
shallower iterations are a much larger share of the total when the last step was the cheap
kind.

The real answer. The shallow iterations are not thrown away. Each one leaves behind a
best move for the root, a killer table, a history table and a set of transposition
entries, and alpha-beta's cost depends almost entirely on being handed good moves first
(slides 51-56, 148). So the shallow searches pay for the deep one. Phase 8 measured the
size of that effect in advance and it is why this phase exists: inside a single
fixed-depth search the transposition move causes **0.63%** of cutoffs, because every hit
arrives with exactly the depth needed and answers the node instead of ordering it; the
moment the table outlives the root it was built for, the same move causes **21.7%**.
Iterative deepening is exactly that structure with the root held still.

--------------------------------------------------------------------------------------
**What this makes possible, and what it costs.**

It is the first search in this project that can stop when it is *out of time* rather than
out of depth, which is what `Budget(time=...)` has meant since Phase 4 and what nothing
has ever used. A fixed-depth search cannot be given a clock: it has no answer at all until
it finishes. An iterative one always has the previous iteration's answer in hand.

That is also the first thing in this project that is not reproducible. How many iterations
fit depends on what else the machine is doing. The rule adopted here, and enforced by
which budget each experiment passes: **a number that gets pinned comes from a node budget
or a depth budget, never from a clock.** `Budget(nodes=...)` aborts at exactly the same
node every time, so an anytime search is still a repeatable one; time is used only where
time is the quantity being measured.

**A half-finished iteration contributes nothing.** The answer is the last *completed*
iteration. The temptation is to keep a partial root score -- the root has examined three
of seven moves, surely the best of those three is worth something -- and it is the classic
bug of this algorithm: the best of three moves is not the value of the position, and
design principle 3 says an optimisation may change how fast the answer arrives and never
what it is. What that costs is measured rather than assumed.

**There is deliberately no early exit on a proven win.** Stopping as soon as a forced mate
is found is standard and would save nothing worth having here, while making "iterative
deepening to depth d returns the depth-d value" only approximately true. The exactness is
worth more than the saving.

--------------------------------------------------------------------------------------
**The parity switch, and why a phase needs one at all.** Phase 5 measured that the root
value rises at every odd depth and falls at every even one in 129 of 129 positions tested,
by about one unanswered three-in-a-row (`THREE = 18`), because an odd depth stops
immediately after the side to move has played and its fresh threats are never answered. A
fixed-depth agent at least has the *same* parity on every move. A time-budgeted one stops
wherever it got to, so its horizon parity varies from move to move -- something nothing in
this project has measured. `parity="even"` answers only from even-depth iterations, which
costs the deepest iteration when it is odd and buys a constant horizon. Both are measured.

--------------------------------------------------------------------------------------
**Three ways an iteration can help the next one, and only two of them can change the move
that gets played.** This is the first phase where that sentence is not "none of them",
which is why each is a separate switch rather than one bundle:

* `root_pv` -- the previous iteration's best move is tried first at the root. Among moves
  of equal value the root keeps the first one it saw, so putting a different move first
  can name a different column. **It changes which of several equally good moves is
  played, never the value.**
* `share_order` -- the killer slots and the history counts carry over. The killers cannot
  reach the root (the root never records a cutoff, so its slots stay empty), but the
  history table is indexed by side and column rather than by ply, so from the second
  iteration it sorts the root too. Same consequence as above, by a different route.
* the transposition table -- shared between iterations by construction, because it is
  created once per *move* in `NegamaxAgent._select_move` and the iteration loop sits
  inside it. It cannot change the move, because the root never probes it. It is not a
  field here for that reason: what the table is worth inside iterative deepening is
  measured the same way Phase 8 measured it, by running with `NO_TABLE` instead.

`NAIVE` below carries neither, which makes it the control: it provably plays the column a
direct search plays, and with the table off it provably costs the exact sum of the
separate searches. Note the qualification -- with the table on it does **not**, and the
difference between the two is precisely what the shared table contributes.

**Note what is NOT reversed here.** The table is shared between the *iterations of one
move* and still thrown away *between moves*. Phase 8 measured why the second half matters:
a table that outlives its root can answer a query with an entry written at a different ply
and return a better value than the search being run would have found, which design
principle 3 forbids even though it is an improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

PARITIES = ("any", "even")


@dataclass(frozen=True, slots=True)
class Deepening:
    """Whether to iterate, and what to carry between iterations. Frozen, like the rest."""

    enabled: bool = False
    root_pv: bool = True
    share_order: bool = True
    parity: str = "any"

    def __post_init__(self) -> None:
        if self.parity not in PARITIES:
            raise ValueError(f"Unknown parity {self.parity!r}; expected one of {PARITIES}.")

    @property
    def name(self) -> str:
        """Carried in every results row. Phase 6 made a node count meaningless without the
        evaluation and the ordering, Phase 8 added the table, and this adds the fourth."""
        if not self.enabled:
            return "direct"
        parts = ["iterative"]
        if self.root_pv:
            parts.append("pv")
        if self.share_order:
            parts.append("order")
        if self.parity != "any":
            parts.append(self.parity)
        return "+".join(parts)

    @property
    def can_change_the_move(self) -> bool:
        """Whether this setting is able to name a different one of several equal moves.

        Stated as a property rather than a comment because it is the boundary Phase 7 and
        Phase 8 both drew by test and this phase is the first to cross it.
        """
        return self.enabled and (self.root_pv or self.share_order)

    def answers(self, depth: int) -> bool:
        """Whether a completed iteration at this depth may supply the answer."""
        return self.parity == "any" or depth % 2 == 0


# Phase 8's search: one search, at the depth it was asked for. Still the default, because
# unlike every optimisation before it this one can change which column comes back.
DIRECT = Deepening()

# The full thing: previous best move first, killers and history carried forward.
ITERATIVE = Deepening(enabled=True)

# Iterate, but learn nothing from it. The control for both halves of the phase -- it plays
# the same column as a direct search, and with no table it costs exactly the sum of the
# shallower ones.
NAIVE = Deepening(enabled=True, root_pv=False, share_order=False)

# Answer only from even depths, so the horizon parity stops varying move to move.
EVEN = Deepening(enabled=True, parity="even")
