"""How one Monte Carlo simulation is run, and how the tree it feeds is grown.

The sixth frozen spec, after `Evaluation`, `Ordering`, `Transposition`, `Deepening` and
`Scout`, and read the same way: a value the agent is configured with, whose `name` is
carried in every row of every results file (thesis point P10). It is named for the
*playout* because that is the switch this phase is really about -- random rollouts against
the light-heuristic ones the greedy agent already implements -- but like `Deepening` it
carries every knob of the search it belongs to, because they are all the same kind of
thing: a decision that could have been made differently and therefore has to be measurable.

--------------------------------------------------------------------------------------
**Why this phase is not like the ten before it.** Every optimisation since Phase 6 has
been the same search made faster, and correctness was one repeated equality: replay the
1,050 frozen minimax values and disagree with none of them. **That gate is useless here.**
MCTS does not compute the minimax value, does not search to a fixed depth, never calls the
evaluation function, and will play different moves. BUILD_PLAN design principle 3 -- an
optimisation must preserve the answer -- **does not apply to it**, which is exactly why
MCTS is a fifth entry in the algorithm picker rather than a sixth checkbox beside
alpha-beta: the five checkboxes are things that provably cannot change the move played,
and this changes what is played.

What replaces the gate is a set of *properties*, and they divide into two kinds that must
never be quoted as if they were one:

* **provable** -- reproducibility (same seed, same budget, same position gives the same
  move and the same counters), the play/undo balance the `Agent` seam checks, the fact
  that a terminal node inside the tree is never rolled out because its value is known
  exactly, and -- only with `solver` on -- that a mate in one is played rather than
  probably played;
* **statistical** -- how often plain UCT finds a mate in one, how often it blocks one, and
  whether the visits concentrate on a move that preserves a forced win. Those are rates
  over a sized sample and are reported as rates.

--------------------------------------------------------------------------------------
**The four steps, and what each one is for.**

1. **Selection.** From the root, walk down by UCB1 -- `mean + C * sqrt(ln N / n)` -- until
   a node is reached that still has a child nobody has tried. The first term is what the
   simulations so far say the move is worth; the second is what is *not yet known* about
   it, and it grows for every sibling that gets tried instead. `C` is the exchange rate
   between the two.
2. **Expansion.** Add exactly one of those children.
3. **Simulation.** Play that position out to the end of the game by a cheap policy and see
   who won.
4. **Backpropagation.** Push the result back up the path, **flipping it at every level**,
   exactly as negamax flips the sign: a game that is a win for me is a loss for whoever
   moved before me. On the [0, 1] reward scale used here the flip is `1 - r` rather than
   `-r`, and the scale is [0, 1] rather than [-1, 1] for one reason only: the C values
   quoted in the literature are for [0, 1], so a measurement here can be compared with
   them without a factor of two in the way.

**Rewards are stored from the point of view of the player who *made* the move into the
node** -- Q(s, a), the value of the move, not of the position. That is what lets the
parent read `child.reward / child.visits` straight out of the child with no negation, and
it is why the flip happens in exactly one place, on the way back up.

--------------------------------------------------------------------------------------
**The rollout policy, and why no new one was written.** `agents/greedy_agent.py` is "take
a win, block a loss, else most central" as a one-reply lookahead, measured in Phase 4
(99.5%/97.6% against random, and an audit showing it never lost a game it could have
saved). It is the standard light-heuristic rollout for this game, so the ablation here is
between two things the project already trusts rather than between a baseline and something
written for the occasion. It is not free: a greedy ply prices roughly fifteen probe
positions against the one a random ply costs, so "more informed" has to *beat* an order of
magnitude more sampling at equal time, and that is a measurement and not an assumption.

**Choosing the move at the root** (`root`). Most-visited is the default rather than
highest-mean: UCB1 spends its visits on the moves it believes in, so the visit count is an
average over the whole search, while a mean computed from three simulations is noise. The
two differ most at small budgets, which is the case the interface actually runs, so the
alternative is a switch and the difference is measured.

**The order the seven children are first tried** (`order`). UCB1 divides by `n`, so it
says nothing about a child with no visits, and the order in which they get their first one
is a real decision that is easy to leave accidental. It is centre-first for the reason
everything else in this project is: of the 69 four-in-a-row lines, 51 pass through column
3 and only 15 through each edge (thesis point P3). `left_to_right` is the control.

**Tree reuse** (`reuse`). After a move is played, the subtree under it is still a valid
search of the new position and throwing it away discards work. The argument against is
*not* Phase 8's argument for the transposition table -- there is no answer to preserve
here, MCTS has no minimax value to protect -- it is that a reused tree makes a move depend
on every move that preceded it, so a position no longer has a cost or an answer of its
own. Every per-position measurement in this project feeds positions in isolation, so the
default is off and the gain is measured rather than assumed.

**The solver** (`solver`). Plain UCT averages; it never proves. A node whose position is
terminal is the one exception that costs nothing -- its value is known exactly, so it is
backed up rather than rolled out, and that is plain UCT rather than an enhancement. Going
further and propagating the proof upwards (a child that is a proven win for the side to
move makes the parent a proven loss; all children proven makes the parent proven) is the
enhancement, it is switchable, and it is what turns "MCTS almost never misses a mate in
one" into "MCTS cannot miss a mate in one".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ROLLOUTS = ("greedy", "random")
ROOTS = ("visits", "mean")
ORDERS = ("center", "left_to_right")

# The textbook exploration constant for rewards on [0, 1].
SQRT2 = math.sqrt(2)


@dataclass(frozen=True, slots=True)
class Playout:
    """Everything about a UCT search that could have been decided differently."""

    rollout: str = "greedy"
    exploration: float = SQRT2
    reuse: bool = False
    root: str = "visits"
    order: str = "center"
    solver: bool = False

    def __post_init__(self) -> None:
        if self.rollout not in ROLLOUTS:
            raise ValueError(f"Unknown rollout {self.rollout!r}; expected one of {ROLLOUTS}.")
        if self.root not in ROOTS:
            raise ValueError(f"Unknown root rule {self.root!r}; expected one of {ROOTS}.")
        if self.order not in ORDERS:
            raise ValueError(f"Unknown order {self.order!r}; expected one of {ORDERS}.")
        if self.exploration < 0.0:
            raise ValueError(f"The exploration constant cannot be negative: {self.exploration}.")

    @property
    def name(self) -> str:
        """The sixth column thesis point P10 requires beside a node count.

        Deliberately independent of this class's own defaults: the rollout is always
        named, so a results file written today still means the same thing if the default
        is ever flipped on the strength of a measurement.
        """
        parts = [f"uct_{self.rollout}"]
        if self.exploration != SQRT2:
            parts.append(f"c{self.exploration:g}")
        if self.reuse:
            parts.append("reuse")
        if self.solver:
            parts.append("solver")
        if self.root != "visits":
            parts.append(self.root)
        if self.order != "center":
            parts.append(self.order)
        return "+".join(parts)


# The default, and what "MCTS" means in the application: light-heuristic rollouts, the
# textbook exploration constant, no memory between moves, no proofs.
UCT = Playout()

# Plain UCT in the textbook sense -- uniformly random playouts. The baseline every rollout
# comparison in this phase is quoted against, in the way `STATIC`, `NO_TABLE`, `DIRECT`
# and `NO_SCOUT` are for their phases.
RANDOM = Playout(rollout="random")

# Keep the subtree under the move that was actually played.
REUSE = Playout(reuse=True)

# Propagate proofs, not just scores.
SOLVER = Playout(solver=True)

# Answer with the move of highest mean reward instead of the most visited one.
MEAN_ROOT = Playout(root="mean")

# The control for the first-visit order: try the columns left to right instead of
# centre-outwards.
LEFT_FIRST = Playout(order="left_to_right")
