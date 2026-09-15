"""What a search may spend, and what it actually spent.

These two records are the halves of one contract. `Budget` is what the caller allows,
`SearchStats` is what the agent reports back. Everything the thesis measures per search
comes out of this file, and nothing in it knows about any particular algorithm: a random
agent and a full alpha-beta search with a transposition table fill in the same record.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import Move


@dataclass(frozen=True, slots=True)
class Budget:
    """Upper bounds on a single search. `None` means "no limit on this axis".

    A budget is a ceiling, not a quota: an agent that has finished may stop early, and an
    agent ignores the axes it has no use for. The baseline agents of Phase 4 ignore all
    three -- they are bounded by their own definition, not by what they are allowed.

    Three axes rather than one union type, because the thesis needs all three and the
    signature of `choose_move` must never change again: node budgets are the primary
    measurement axis (BUILD_PLAN 6.1), depth is what a fixed-depth minimax takes, and
    time is the only honest common currency between alpha-beta and MCTS.
    """

    nodes: int | None = None
    time: float | None = None  # seconds
    depth: int | None = None

    def __str__(self) -> str:
        limits = []
        if self.nodes is not None:
            limits.append(f"{self.nodes} nodes")
        if self.time is not None:
            limits.append(f"{self.time:g}s")
        if self.depth is not None:
            limits.append(f"depth {self.depth}")
        return ", ".join(limits) if limits else "unlimited"


UNLIMITED = Budget()


@dataclass(slots=True)
class SearchStats:
    """One record per call to `choose_move`.

    Mutable on purpose: a recursive search increments the counters in place rather than
    threading return values through every call. A node is a position reached by making a
    move, which is the same unit every later agent counts, so node numbers stay comparable
    from the greedy agent all the way to negascout.
    """

    nodes: int = 0
    leaves: int = 0  # nodes the search stopped at: terminal, or out of depth
    depth: int = 0
    cutoffs: int = 0
    elapsed: float = 0.0  # seconds, filled in by Agent.choose_move
    move: Move | None = None
    value: int | None = None  # root score on the core.types.Score scale

    # --------------------------------------------------------------- ordering quality
    #
    # Phase 7. A node count says how much work the search did; these say how well it was
    # steered, which is a different question and the one move ordering is actually about.
    # A cutoff found on the first move tried is a node where the ordering was perfect --
    # every other move in it was skipped. The share of those is the standard measure of
    # ordering quality, and `cut_index_total` gives the mean rank of the move that cut,
    # which is the same statistic without the all-or-nothing rounding.
    #
    # They are reported separately from the node count on purpose: the two are correlated
    # but not the same thing, and Phase 7 found positions where the ordering quality rises
    # and the node count rises with it.
    first_cutoffs: int = 0  # cutoffs that happened on the first move tried
    cut_index_total: int = 0  # sum of the 1-based rank of every cutting move
    killer_cutoffs: int = 0  # cutoffs caused by a move the killer table had promoted

    @property
    def first_cutoff_rate(self) -> float:
        """Share of cut nodes that cut immediately. 1.0 would be a perfect ordering."""
        return self.first_cutoffs / self.cutoffs if self.cutoffs else 0.0

    @property
    def mean_cut_index(self) -> float:
        """Average rank of the move that caused a cutoff. 1.0 would be perfect."""
        return self.cut_index_total / self.cutoffs if self.cutoffs else 0.0
