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

    `nodes` and `time` did nothing at all until Phase 9, because a fixed-depth search has
    no answer to give until it finishes and so cannot be stopped part way. An iterative
    one always holds the previous iteration's answer, which is what makes either of them
    enforceable. The two are not interchangeable: **a node budget aborts at the same node
    every time and a time budget does not**, so every pinned number in this thesis comes
    from `nodes` or `depth` and `time` is used only where time is the quantity being
    measured.
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

    # ------------------------------------------------------------- iterative deepening
    #
    # Phase 9. `depth` stops being a setting and becomes a result: it is the depth of the
    # iteration the answer came from, which under a clock is whatever fitted. `depth_cap`
    # is what was allowed, so the pair says how much of the budget the position used.
    #
    # `aborted` is the honest half of a time budget. An iteration that ran out of time
    # contributes nothing -- its partial root score is discarded -- so this flag is the
    # only trace it leaves, and without it a search that stopped early is indistinguishable
    # from one that simply finished.
    iterations: int = 0  # iterations that completed
    depth_cap: int = 0  # the deepest iteration that was allowed
    aborted: bool = False  # an iteration was cut off by the budget and thrown away

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
    tt_move_cutoffs: int = 0  # cutoffs caused by the move the table remembered

    # --------------------------------------------------------- transposition table
    #
    # Phase 8, and reported for the same reason as the ordering-quality figures above: a
    # node count says how much work was avoided, not where the saving came from. These
    # are copied off the table when the search finishes, so they cost nothing per node.
    #
    # `tt_returns` is the one that matters most. A probe that hits is not yet a saving --
    # the entry may be a bound that does not settle the current window, in which case all
    # it contributes is a move to try first. A *return* is a whole subtree not searched.
    tt_probes: int = 0
    tt_hits: int = 0
    tt_returns: int = 0  # probes that answered the node outright
    tt_stores: int = 0
    tt_rewrites: int = 0  # the same position stored again, its first entry being a bound
    tt_collisions: int = 0  # a different position evicted (fixed-size table only)
    tt_refused: int = 0  # stores dropped: table full, or beaten on depth
    tt_entries: int = 0
    tt_peak: int = 0
    tt_capacity: int = 0

    @property
    def first_cutoff_rate(self) -> float:
        """Share of cut nodes that cut immediately. 1.0 would be a perfect ordering."""
        return self.first_cutoffs / self.cutoffs if self.cutoffs else 0.0

    @property
    def mean_cut_index(self) -> float:
        """Average rank of the move that caused a cutoff. 1.0 would be perfect."""
        return self.cut_index_total / self.cutoffs if self.cutoffs else 0.0

    @property
    def tt_hit_rate(self) -> float:
        """Share of probes that found the position already in the table."""
        return self.tt_hits / self.tt_probes if self.tt_probes else 0.0

    @property
    def tt_return_rate(self) -> float:
        """Share of probes that ended the node there and then."""
        return self.tt_returns / self.tt_probes if self.tt_probes else 0.0

    @property
    def tt_fill(self) -> float:
        return self.tt_entries / self.tt_capacity if self.tt_capacity else 0.0

    def take_table(self, table) -> None:
        """Copy a finished search's table counters in. None when the table is off."""
        if table is None:
            return
        self.tt_probes = table.probes
        self.tt_hits = table.hits
        self.tt_stores = table.stores
        self.tt_rewrites = table.rewrites
        self.tt_collisions = table.collisions
        self.tt_refused = table.overflows
        self.tt_entries = table.entries
        self.tt_peak = table.peak
        self.tt_capacity = table.capacity
