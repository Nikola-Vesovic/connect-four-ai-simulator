"""What one more ply of plain minimax costs, and what it buys.

Two measurements, both of which the rest of the thesis is calibrated against.

**Cost.** Plain minimax has no cutoffs, so it visits its entire tree. The node counts
below are the baseline alpha-beta is measured against in Phase 6, and they have a
property worth exploiting: with no pruning the count cannot depend on what the evaluation
function returns, which a test in `tests/test_minimax.py` proves. So the counts are taken
with the zero evaluation, where a node costs a fraction of what it otherwise would, and
the depths that would be unreachable otherwise come within reach. Wall-clock is reported
for both evaluations, because the *time* very much does depend on it.

**Benefit.** Score against the greedy agent, depth by depth. The expectation on record in
BUILD_PLAN is a rough tie at depth 2 -- greedy is a two-ply minimax with a trivial
evaluation, so at that depth the two differ only in what they think a quiet position is
worth -- and an advantage from depth 3. That expectation is being tested here, not
assumed, and the curve is the result of the phase either way.

Both agents are deterministic, so every game from the empty board would be the same game.
Each pair of games therefore starts from a random four-ply opening, shared between the two
games of the pair so that each opening is played once from each seat.

**Phase 6 adds what alpha-beta does to those numbers**, and with it a rule that holds for
every node count from here on: a pruned search's tree depends on the evaluation function
and on the move ordering, so no node count is quotable without naming both. Plain minimax
was exempt -- with no cutoffs there is nothing for either to steer -- which is exactly why
the baseline above could be taken with the cheap evaluation.

**Phase 7 adds the ordering sweep.** Phase 6 held the ordering at centre-first so it had
one thing to vary; this one varies it, against exactly those numbers. Three measurements,
because any one of them alone would mislead: the empty board (which Phase 3 showed is
alpha-beta's best case), the 200 reference positions, and complete games. Node counts and
seconds are both reported, because killers and history add per-node bookkeeping and the
two need not move together.

**Phase 8 adds the transposition table**, measured the same three ways and with one extra
measurement in front of them: how many of the positions in the tree are repeats at all,
counted directly, before any search is involved. That number is the ceiling, and quoting a
gain without it would leave the reader unable to tell a good result from a poor one.

**Phase 10 adds NegaScout**, measured the same three ways and with two more beside them,
because this is the first optimisation whose value depends on another one. Slide 148 ties
its performance directly to the move ordering, so the ordering is swept against it; and
Phase 9 measured that iterative deepening produces a much better first move, which is
exactly the condition it needs, so the deepening switch is swept against it too. Every one
of those rows carries the *scouting* column as well, which is the fifth thing thesis point
P10 now requires beside a node count.

    python -m experiments.scaling --nodes
    python -m experiments.scaling --curve
    python -m experiments.scaling --pruning
    python -m experiments.scaling --positions
    python -m experiments.scaling --ordering
    python -m experiments.scaling --ordering-games
    python -m experiments.scaling --density
    python -m experiments.scaling --table
    python -m experiments.scaling --table-empty
    python -m experiments.scaling --table-games
    python -m experiments.scaling --scout
    python -m experiments.scaling --scout-empty
    python -m experiments.scaling --scout-games
    python -m experiments.scaling --scout-ordering
    python -m experiments.scaling --scout-deepening
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

from agents.alphabeta import NegamaxAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from experiments import reference
from experiments.match import Tally, play_match
from search.deepening import DIRECT, EVEN, ITERATIVE, NAIVE, Deepening
from search.evaluation import DEFAULT, ZERO
from search.ordering import DYNAMIC, LEFT, RANDOM, STATIC, Ordering
from search.scouting import DEEP, NO_SCOUT, SCOUT, WIDE, Scout
from search.stats import Budget
from search.transposition import ARRAY, NO_TABLE, PERSISTENT, TABLE, WRITE_ONLY, Transposition

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
NODES_CSV = RESULTS / "minimax_nodes_by_depth.csv"
CURVE_CSV = RESULTS / "minimax_vs_greedy.csv"
PRUNING_CSV = RESULTS / "alphabeta_nodes_by_depth.csv"
POSITIONS_CSV = RESULTS / "alphabeta_by_position.csv"
ORDERING_CSV = RESULTS / "ordering_by_position.csv"
ORDERING_EMPTY_CSV = RESULTS / "ordering_on_empty_board.csv"
ORDERING_GAMES_CSV = RESULTS / "ordering_by_game.csv"
DENSITY_CSV = RESULTS / "transposition_density.csv"
TABLE_CSV = RESULTS / "transposition_by_position.csv"
TABLE_EMPTY_CSV = RESULTS / "transposition_on_empty_board.csv"
TABLE_GAMES_CSV = RESULTS / "transposition_by_game.csv"
DEEPENING_CSV = RESULTS / "deepening_by_position.csv"
DEEPENING_EMPTY_CSV = RESULTS / "deepening_on_empty_board.csv"
DEEPENING_GAMES_CSV = RESULTS / "deepening_by_game.csv"
DEEPENING_CLOCK_CSV = RESULTS / "deepening_under_a_clock.csv"
SCOUT_CSV = RESULTS / "scout_by_position.csv"
SCOUT_EMPTY_CSV = RESULTS / "scout_on_empty_board.csv"
SCOUT_GAMES_CSV = RESULTS / "scout_by_game.csv"
SCOUT_ORDERING_CSV = RESULTS / "scout_by_ordering.csv"
SCOUT_DEEPENING_CSV = RESULTS / "scout_under_deepening.csv"

# The tables compared. The first is Phase 7's search and every gain is quoted against it.
# `write_only` is the controlled experiment: it keeps the table exactly as the real one
# does but never reads an answer out of it, so it provably searches the same tree and
# every second of difference between it and `no_table` is bookkeeping and nothing else.
# The small array is in to show what a table too small for the job costs.
SMALL_ARRAY = Transposition(kind="array", capacity=1 << 12)
TRANSPOSITIONS: tuple[Transposition, ...] = (
    NO_TABLE,
    TABLE,
    ARRAY,
    SMALL_ARRAY,
    WRITE_ONLY,
)

# The transposition setting every Phase 6 and Phase 7 node count was measured under, kept
# as the CSV label so that old and new result files line up (BUILD_PLAN thesis point P10:
# a node count is not a measurement unless the evaluation, the ordering and the table that
# produced it are named beside it).
TABLE_OFF = NO_TABLE.name

# The orderings compared. The first is Phase 6's and every gain is quoted against it; the
# last two are the failure modes slide 148 warns about, measured rather than repeated,
# each also shown with the learned parts switched on to see how much they win back.
ORDERINGS: tuple[Ordering, ...] = (
    STATIC,
    Ordering(killers=True, history=False),
    Ordering(killers=False, history=True),
    DYNAMIC,
    LEFT,
    Ordering(base="left_to_right", killers=True, history=True),
    RANDOM,
    Ordering(base="random", killers=True, history=True),
)

# The ordering every Phase 6 node count was measured under, kept as the CSV label it was
# written with so old and new result files line up.
ORDERING = STATIC.name

# The deepening settings compared. The first is Phase 8's search and every gain is quoted
# against it. The two in the middle take the carried state apart: the root principal
# variation on its own, and the killer/history tables on their own, so the bundle is never
# presented as if each half contributed what the pair does. `naive` carries neither and is
# the control that cannot change the move played.
PV_ONLY = Deepening(enabled=True, root_pv=True, share_order=False)
ORDER_ONLY = Deepening(enabled=True, root_pv=False, share_order=True)
DEEPENINGS: tuple[Deepening, ...] = (DIRECT, NAIVE, PV_ONLY, ORDER_ONLY, ITERATIVE, EVEN)

# The scouting settings compared. The first is Phase 9's search and every gain is quoted
# against it. `negascout+wide` is the control for the re-search window -- the classic bug
# of this algorithm -- and must return identical values while visiting different nodes;
# `negascout+from3` leaves the degenerate depths alone, where the children of a node are
# horizon nodes and a null window has nothing it can cut.
SCOUTS: tuple[Scout, ...] = (NO_SCOUT, SCOUT, WIDE, DEEP)

# The budget every node count before Phase 9 was measured under. It is the fourth column
# thesis point P10 now requires beside the evaluation, the ordering and the table: a
# search that stops on a clock has a node count that depends on the machine, so every
# number in these files is produced by a depth budget and says so.
BUDGET = "depth"


def standard_error(tally: Tally) -> float:
    """How far the measured score could reasonably be from the true one, one sigma.

    Draws count a half, so the per-game score is not a Bernoulli variable; this uses its
    actual variance rather than p(1-p), which would understate a drawish pairing.
    """
    if tally.games < 2:
        return 0.0
    outcomes = [1.0] * tally.wins + [0.5] * tally.draws + [0.0] * tally.losses
    mean = sum(outcomes) / len(outcomes)
    variance = sum((value - mean) ** 2 for value in outcomes) / (len(outcomes) - 1)
    return math.sqrt(variance / len(outcomes))


def nodes_by_depth(max_depth: int, timed_depth: int) -> list[dict]:
    """Tree size from the empty board, depth by depth.

    `timed_depth` is how deep the full evaluation is also timed. The node counts are the
    same either way; only the seconds differ, and above a few plies the full evaluation
    takes long enough that measuring it stops being informative and starts being a wait.
    """
    rows = []
    for depth in range(1, max_depth + 1):
        started = time.perf_counter()
        _, cheap = MinimaxAgent(depth, evaluation=ZERO).choose_move(Bitboard())
        zero_seconds = time.perf_counter() - started

        full_seconds = ""
        if depth <= timed_depth:
            started = time.perf_counter()
            _, full = MinimaxAgent(depth, evaluation=DEFAULT).choose_move(Bitboard())
            full_seconds = round(time.perf_counter() - started, 3)
            assert full.nodes == cheap.nodes and full.leaves == cheap.leaves

        rows.append(
            {
                "depth": depth,
                "nodes": cheap.nodes,
                "leaves": cheap.leaves,
                "full_tree_nodes": sum(7**level for level in range(1, depth + 1)),
                "zero_eval_seconds": round(zero_seconds, 3),
                "full_eval_seconds": full_seconds,
            }
        )
        print(
            f"depth {depth}: {cheap.nodes:>12,} nodes  {cheap.leaves:>12,} leaves  "
            f"{zero_seconds:8.2f}s zero  "
            f"{full_seconds if full_seconds != '' else '-':>9} full",
            flush=True,
        )
    return rows


def score_vs_greedy(plan, seed: int, opening_plies: int) -> list[dict]:
    rows = []
    for depth, games, evaluation in plan:
        started = time.perf_counter()
        agent = MinimaxAgent(depth, evaluation=evaluation)
        match = play_match(
            agent,
            GreedyAgent(),
            games=games,
            seed=seed,
            opening_plies=opening_plies,
        )
        rows.append(
            {
                "agent": agent.name,
                "depth": depth,
                "evaluation": evaluation.name,
                "games": games,
                "opening_plies": opening_plies,
                "score_first": round(match.a_first.score, 4),
                "error_first": round(standard_error(match.a_first), 4),
                "score_second": round(match.b_first.score, 4),
                "error_second": round(standard_error(match.b_first), 4),
                "score_overall": round(match.overall.score, 4),
                "wins": match.overall.wins,
                "draws": match.overall.draws,
                "losses": match.overall.losses,
                "nodes_per_move": round(match.nodes_a / match.moves_a, 1),
                "mean_plies": round(match.mean_plies, 2),
                "seconds": round(time.perf_counter() - started, 1),
                "digest": match.digest(),
            }
        )
        row = rows[-1]
        print(
            f"{row['agent']:<18} first {row['score_first']:.1%} +-{row['error_first']:.1%}   "
            f"second {row['score_second']:.1%} +-{row['error_second']:.1%}   "
            f"overall {row['score_overall']:.1%}   "
            f"{row['wins']}-{row['draws']}-{row['losses']}   "
            f"{row['nodes_per_move']:,} nodes/move   {row['seconds']}s",
            flush=True,
        )
    return rows


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def pruning_by_depth(max_depth: int, unpruned_depth: int, max_seconds: float) -> list[dict]:
    """The headline of Phase 6: how much of the tree alpha-beta never looks at.

    One row per (depth, evaluation), because from this phase on the evaluation is part of
    the measurement. The unpruned column is measured with the zero evaluation and with
    pruning switched off, and cross-checked against the committed Phase 5 baseline -- it
    has to agree, since a search without cutoffs cannot be steered by anything.

    Depths past `unpruned_depth` have no unpruned counterpart, on purpose: the whole point
    is that minimax cannot get there. Those rows answer "how deep does pruning let it go",
    and the sweep stops when one search passes `max_seconds`.
    """
    baseline = {int(row["depth"]): row for row in csv.DictReader(NODES_CSV.open())}
    rows = []

    for depth in range(1, max_depth + 1):
        unpruned_nodes = unpruned_seconds = None
        if depth <= unpruned_depth:
            started = time.perf_counter()
            _, plain = NegamaxAgent(depth, pruning=False, evaluation=ZERO, transposition=NO_TABLE).choose_move(Bitboard())
            unpruned_seconds = time.perf_counter() - started
            unpruned_nodes = plain.nodes
            if depth in baseline:
                recorded = int(baseline[depth]["nodes"])
                assert unpruned_nodes == recorded, (
                    f"depth {depth}: unpruned negamax visited {unpruned_nodes} nodes, "
                    f"but the Phase 5 minimax baseline says {recorded}"
                )

        slowest = 0.0
        for evaluation in (ZERO, DEFAULT):
            started = time.perf_counter()
            _, stats = NegamaxAgent(depth, pruning=True, evaluation=evaluation, transposition=NO_TABLE, scouting=NO_SCOUT).choose_move(
                Bitboard()
            )
            seconds = time.perf_counter() - started
            slowest = max(slowest, seconds)
            rows.append(
                {
                    "depth": depth,
                    "evaluation": evaluation.name,
                    "ordering": ORDERING,
                    "unpruned_nodes": unpruned_nodes if unpruned_nodes is not None else "",
                    "alphabeta_nodes": stats.nodes,
                    "alphabeta_leaves": stats.leaves,
                    "cutoffs": stats.cutoffs,
                    "node_ratio": (
                        round(stats.nodes / unpruned_nodes, 5) if unpruned_nodes else ""
                    ),
                    "speedup": (
                        round(unpruned_nodes / stats.nodes, 1) if unpruned_nodes else ""
                    ),
                    "ebf_unpruned": (
                        round(unpruned_nodes ** (1 / depth), 3) if unpruned_nodes else ""
                    ),
                    "ebf_alphabeta": round(stats.nodes ** (1 / depth), 3),
                    "value": stats.value,
                    "alphabeta_seconds": round(seconds, 3),
                    "unpruned_seconds": (
                        round(unpruned_seconds, 3) if unpruned_seconds is not None else ""
                    ),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2} {evaluation.name:<5}  "
                f"unpruned {str(row['unpruned_nodes'] or '-'):>12}  "
                f"alpha-beta {stats.nodes:>10,}  "
                f"ratio {str(row['node_ratio'] or '-'):>8}  "
                f"ebf {row['ebf_alphabeta']:>6}  "
                f"cutoffs {stats.cutoffs:>9,}  {seconds:8.2f}s",
                flush=True,
            )
        if slowest > max_seconds:
            print(f"stopping: depth {depth} took {slowest:.1f}s, over the {max_seconds:g}s cap")
            break
    return rows


def pruning_by_position(max_depth: int) -> list[dict]:
    """The same ratio over real positions instead of the empty board.

    The empty board is one position, and a flattering one: it is symmetric, and centre
    ordering is close to perfect there. The reference set is 200 positions spread across
    the stages of a game, and it already stores the node count plain minimax needed for
    every one of them, so replaying it with alpha-beta costs nothing extra and gives a
    ratio with a spread rather than a single number.
    """
    data = reference.load()
    by_depth: dict[int, list[tuple[int, int]]] = {}
    for record in data["records"]:
        if record["depth"] > max_depth:
            continue
        agent = NegamaxAgent(record["depth"], pruning=True, evaluation=DEFAULT, transposition=NO_TABLE, scouting=NO_SCOUT)
        _, stats = agent.choose_move(Bitboard.from_move_string(record["moves"]))
        assert stats.value == record["value"], f"value moved on {record['moves']}"
        by_depth.setdefault(record["depth"], []).append((record["nodes"], stats.nodes))

    rows = []
    for depth, pairs in sorted(by_depth.items()):
        plain = sum(pair[0] for pair in pairs)
        pruned = sum(pair[1] for pair in pairs)
        ratios = sorted(after / before for before, after in pairs)
        rows.append(
            {
                "depth": depth,
                "evaluation": DEFAULT.name,
                "ordering": ORDERING,
                "transposition": TABLE_OFF,
                "positions": len(pairs),
                "minimax_nodes": plain,
                "alphabeta_nodes": pruned,
                "node_ratio": round(pruned / plain, 5),
                "speedup": round(plain / pruned, 2),
                "ratio_best": round(ratios[0], 5),
                "ratio_median": round(ratios[len(ratios) // 2], 5),
                "ratio_worst": round(ratios[-1], 5),
            }
        )
        row = rows[-1]
        print(
            f"depth {depth}: {len(pairs):>3} positions  "
            f"{plain:>11,} -> {pruned:>9,} nodes  "
            f"ratio {row['node_ratio']:.4f}  ({row['speedup']:.1f}x)  "
            f"per-position best {row['ratio_best']:.3f} / "
            f"median {row['ratio_median']:.3f} / worst {row['ratio_worst']:.3f}",
            flush=True,
        )
    return rows


def ordering_on_empty_board(depths: tuple[int, ...]) -> list[dict]:
    """The same sweep on the one position everybody benchmarks, for the contrast.

    Kept separate from the position sweep and never quoted on its own. The empty board is
    symmetric and centre-first ordering is nearly perfect on it -- thesis point P11 -- so
    it is where a *learned* ordering has the least to add, and it is the one place in this
    phase where the numbers can come out worse.
    """
    rows = []
    for depth in depths:
        baseline = None
        for spec in ORDERINGS:
            agent = NegamaxAgent(depth, ordering=spec, evaluation=DEFAULT, transposition=NO_TABLE, scouting=NO_SCOUT, seed=2026)
            started = time.perf_counter()
            _, stats = agent.choose_move(Bitboard())
            seconds = time.perf_counter() - started
            if baseline is None:
                baseline = stats.nodes
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": spec.name,
                    "transposition": TABLE_OFF,
                    "nodes": stats.nodes,
                    "leaves": stats.leaves,
                    "cutoffs": stats.cutoffs,
                    "gain": round(baseline / stats.nodes, 3),
                    "first_cutoff_rate": round(stats.first_cutoff_rate, 4),
                    "mean_cut_index": round(stats.mean_cut_index, 3),
                    "killer_cutoff_rate": round(
                        stats.killer_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    "value": stats.value,
                    "seconds": round(seconds, 3),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2}  {spec.name:<30} {stats.nodes:>10,} nodes  "
                f"{row['gain']:>5}x  first-cut {row['first_cutoff_rate']:>6.1%}  "
                f"{seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def ordering_by_position(depths: tuple[int, ...]) -> list[dict]:
    """The gate of Phase 7: the same sweep over the 200 frozen reference positions.

    Every value is checked against the stored one as it goes, so the run is simultaneously
    the measurement and a correctness check -- an ordering that changed an answer would
    stop the sweep rather than quietly produce a better-looking number.
    """
    data = reference.load()
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        baseline = None
        for spec in ORDERINGS:
            agent = NegamaxAgent(depth, ordering=spec, evaluation=DEFAULT, transposition=NO_TABLE, scouting=NO_SCOUT, seed=2026)
            nodes = leaves = cutoffs = first = index_total = killer = 0
            seconds = 0.0
            for record in records:
                board = Bitboard.from_move_string(record["moves"])
                started = time.perf_counter()
                _, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                assert stats.value == record["value"], (
                    f"{spec.name} at depth {depth} changed the value of {record['moves']!r}"
                )
                nodes += stats.nodes
                leaves += stats.leaves
                cutoffs += stats.cutoffs
                first += stats.first_cutoffs
                index_total += stats.cut_index_total
                killer += stats.killer_cutoffs
            if baseline is None:
                baseline = nodes
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": spec.name,
                    "transposition": TABLE_OFF,
                    "positions": len(records),
                    "nodes": nodes,
                    "leaves": leaves,
                    "cutoffs": cutoffs,
                    "gain": round(baseline / nodes, 3),
                    "ebf": round(nodes ** (1 / depth) / len(records) ** (1 / depth), 4),
                    "first_cutoff_rate": round(first / cutoffs, 4) if cutoffs else "",
                    "mean_cut_index": round(index_total / cutoffs, 3) if cutoffs else "",
                    "killer_cutoff_rate": round(killer / cutoffs, 4) if cutoffs else "",
                    "seconds": round(seconds, 2),
                    "nodes_per_second": round(nodes / seconds) if seconds else "",
                }
            )
            row = rows[-1]
            print(
                f"depth {depth}  {spec.name:<30} {nodes:>10,} nodes  {row['gain']:>5}x  "
                f"first-cut {row['first_cutoff_rate']:>6.1%}  "
                f"mean idx {row['mean_cut_index']:>5}  {seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def ordering_by_game(depth: int, games: int, seed: int, opening_plies: int) -> list[dict]:
    """What a whole game costs, which is what somebody using the window actually pays.

    Both seats use the same ordering and everything both of them spend is counted. A game
    is not a sample of one position: it walks from an opening, where alpha-beta does well,
    into an endgame where there is hardly any tree left to prune, so the average over a
    game is a different number from the average over a set of positions and both are worth
    having.
    """
    rng = random.Random(seed)
    openings = [
        [rng.randrange(7) for _ in range(opening_plies)] for _ in range(games)
    ]
    rows = []
    baseline = None
    for spec in ORDERINGS:
        agent = NegamaxAgent(depth, ordering=spec, evaluation=DEFAULT, transposition=NO_TABLE, scouting=NO_SCOUT, seed=seed)
        nodes = cutoffs = first = index_total = plies = 0
        seconds = 0.0
        for opening in openings:
            board = Bitboard()
            for col in opening:
                if board.is_terminal() or not board.can_play(col):
                    break
                board.play(col)
            while not board.is_terminal():
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                nodes += stats.nodes
                cutoffs += stats.cutoffs
                first += stats.first_cutoffs
                index_total += stats.cut_index_total
                plies += 1
                board.play(move)
        if baseline is None:
            baseline = nodes
        rows.append(
            {
                "depth": depth,
                "evaluation": DEFAULT.name,
                "ordering": spec.name,
                "transposition": TABLE_OFF,
                "games": games,
                "opening_plies": opening_plies,
                "moves": plies,
                "nodes": nodes,
                "nodes_per_move": round(nodes / plies, 1),
                "cutoffs": cutoffs,
                "gain": round(baseline / nodes, 3),
                "first_cutoff_rate": round(first / cutoffs, 4) if cutoffs else "",
                "mean_cut_index": round(index_total / cutoffs, 3) if cutoffs else "",
                "seconds": round(seconds, 2),
                "seconds_per_move": round(seconds / plies, 4),
            }
        )
        row = rows[-1]
        print(
            f"{spec.name:<30} {nodes:>10,} nodes over {plies} moves  {row['gain']:>5}x  "
            f"first-cut {row['first_cutoff_rate']:>6.1%}  {seconds:7.2f}s",
            flush=True,
        )
    return rows


# ------------------------------------------------------------------ Phase 8: the table


def transposition_density(depths: tuple[int, ...], positions: int) -> list[dict]:
    """The ceiling, measured before any search is involved.

    Walks the full fixed-depth tree -- the shape plain minimax visits -- and counts nodes
    against DISTINCT positions. A transposition table can never save more than the gap,
    and knowing the gap is what turns the phase's result into a number that can be judged
    rather than merely reported.

    Two honest qualifications go with every figure here. It is the ceiling on a *minimax*
    tree, and alpha-beta already declines to visit most of that tree, so the repeats
    actually available in a pruned search are far fewer. And a repeat still costs one node
    visit even when its whole subtree is skipped, so the ratio is an upper bound that even
    a perfect table cannot reach.
    """
    data = reference.load()
    samples = [record["moves"] for record in data["records"] if record["depth"] == 1]
    samples = samples[:positions]
    rows = []
    for depth in depths:
        for label, chosen in (("empty board", [""]), (f"{len(samples)} positions", samples)):
            nodes = distinct = 0
            started = time.perf_counter()
            for start in chosen:
                board = Bitboard.from_move_string(start)
                seen: set[int] = set()
                nodes += _walk(board, depth, seen)
                distinct += len(seen)
            rows.append(
                {
                    "depth": depth,
                    "sample": label,
                    "positions": len(chosen),
                    "nodes": nodes,
                    "distinct": distinct,
                    "repeats": nodes - distinct,
                    "ceiling": round(nodes / distinct, 3),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2}  {label:<14} {nodes:>12,} nodes  "
                f"{distinct:>11,} distinct  ceiling {row['ceiling']:>7.2f}x",
                flush=True,
            )
    return rows


def _walk(board, depth: int, seen: set) -> int:
    """Every node of the full tree below `board`, collecting distinct position keys."""
    nodes = 0
    for col in board.legal_moves():
        board.play(col)
        nodes += 1
        seen.add(board.key())
        if depth > 1 and not board.is_terminal():
            nodes += _walk(board, depth - 1, seen)
        board.undo()
    return nodes


def _table_row(stats, spec: Transposition) -> dict:
    """The transposition counters, as their own measurement rather than a node count.

    `hit` and `return` are deliberately separate. A probe that finds the position is not
    yet a saving: the entry may be a bound that does not settle this node's window, in
    which case all it supplies is a move to try first. A *return* is a whole subtree never
    searched. The gap between the two rates is how often the table was consulted usefully
    but could not finish the job.
    """
    return {
        "tt_probes": stats.tt_probes,
        "tt_hits": stats.tt_hits,
        "tt_hit_rate": round(stats.tt_hit_rate, 4),
        "tt_returns": stats.tt_returns,
        "tt_return_rate": round(stats.tt_return_rate, 4),
        "tt_stores": stats.tt_stores,
        "tt_rewrites": stats.tt_rewrites,
        "tt_collisions": stats.tt_collisions,
        "tt_refused": stats.tt_refused,
        "tt_entries": stats.tt_entries,
        "tt_peak": stats.tt_peak,
        "tt_capacity": stats.tt_capacity if spec.enabled else "",
        "tt_fill": round(stats.tt_fill, 5) if spec.enabled else "",
    }


def _accumulate(total: dict, stats) -> None:
    for field in (
        "nodes", "leaves", "cutoffs", "first_cutoffs", "cut_index_total", "killer_cutoffs",
        "tt_move_cutoffs", "tt_probes", "tt_hits", "tt_returns", "tt_stores", "tt_rewrites",
        "tt_collisions", "tt_refused", "scout_searches", "researches",
    ):
        total[field] = total.get(field, 0) + getattr(stats, field)
    for field in ("tt_entries", "tt_peak"):
        total[field] = max(total.get(field, 0), getattr(stats, field))
    total["tt_capacity"] = stats.tt_capacity


def transposition_by_position(depths: tuple[int, ...]) -> list[dict]:
    """The gate of Phase 8: the same sweep over the 200 frozen reference positions.

    Every value is checked against the stored one as it goes, so the run is simultaneously
    the measurement and a correctness check. A table that changed an answer would stop the
    sweep rather than quietly produce a better-looking number, which matters more here
    than it did in Phase 7: move ordering structurally *cannot* change a value, while a
    transposition table very much can if the bound logic is wrong.
    """
    data = reference.load()
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        baseline = baseline_seconds = None
        for spec in TRANSPOSITIONS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=spec,
                scouting=NO_SCOUT, seed=2026,
            )
            total: dict = {}
            seconds = 0.0
            for record in records:
                board = Bitboard.from_move_string(record["moves"])
                started = time.perf_counter()
                _, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                assert stats.value == record["value"], (
                    f"{spec.name} at depth {depth} changed the value of {record['moves']!r}"
                )
                _accumulate(total, stats)
            nodes, cutoffs = total["nodes"], total["cutoffs"]
            if baseline is None:
                baseline, baseline_seconds = nodes, seconds
            if spec is WRITE_ONLY:
                # The control. It maintains the table and never acts on it, so it must
                # search exactly the tree the baseline searched -- asserted, not hoped.
                assert nodes == baseline, f"write_only changed the tree at depth {depth}"
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": spec.name,
                    "positions": len(records),
                    "nodes": nodes,
                    "leaves": total["leaves"],
                    "cutoffs": cutoffs,
                    "gain": round(baseline / nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "ebf": round((nodes / len(records)) ** (1 / depth), 4),
                    "first_cutoff_rate": round(total["first_cutoffs"] / cutoffs, 4),
                    "mean_cut_index": round(total["cut_index_total"] / cutoffs, 3),
                    "killer_cutoff_rate": round(total["killer_cutoffs"] / cutoffs, 4),
                    "tt_move_cutoff_rate": round(total["tt_move_cutoffs"] / cutoffs, 4),
                    **_table_row(_Totals(total), spec),
                    "seconds": round(seconds, 2),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth}  {spec.name:<22} {nodes:>10,} nodes  {row['gain']:>5}x  "
                f"hit {row['tt_hit_rate']:>6.1%}  ret {row['tt_return_rate']:>6.1%}  "
                f"killer-cut {row['killer_cutoff_rate']:>6.1%}  "
                f"tt-cut {row['tt_move_cutoff_rate']:>5.1%}  {seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


class _Totals:
    """Reads like a `SearchStats` so `_table_row` can be shared between one search and a
    sweep over two hundred of them."""

    def __init__(self, total: dict) -> None:
        self.__dict__.update(total)

    @property
    def tt_hit_rate(self) -> float:
        return self.tt_hits / self.tt_probes if self.tt_probes else 0.0

    @property
    def tt_return_rate(self) -> float:
        return self.tt_returns / self.tt_probes if self.tt_probes else 0.0

    @property
    def tt_fill(self) -> float:
        return self.tt_entries / self.tt_capacity if self.tt_capacity else 0.0


def transposition_on_empty_board(depths: tuple[int, ...]) -> list[dict]:
    """The position Phase 7 lost on, which is the one this phase expects to win on.

    The reasoning was written down before the measurement: the empty board is the most
    symmetric position there is, so it is where the same arrangement of discs is reachable
    by the most different move orders, and therefore where a table has the most to find.
    It is also the position everybody benchmarks, so quoting only this one would repeat
    Phase 6's mistake in the other direction -- it is reported beside the position sweep,
    never instead of it.
    """
    rows = []
    for depth in depths:
        baseline = baseline_seconds = None
        for spec in TRANSPOSITIONS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=spec,
                scouting=NO_SCOUT, seed=2026,
            )
            started = time.perf_counter()
            _, stats = agent.choose_move(Bitboard())
            seconds = time.perf_counter() - started
            if baseline is None:
                baseline, baseline_seconds = stats.nodes, seconds
                baseline_value = stats.value
            assert stats.value == baseline_value, f"{spec.name} changed the value at {depth}"
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": spec.name,
                    "nodes": stats.nodes,
                    "leaves": stats.leaves,
                    "cutoffs": stats.cutoffs,
                    "gain": round(baseline / stats.nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "first_cutoff_rate": round(stats.first_cutoff_rate, 4),
                    "mean_cut_index": round(stats.mean_cut_index, 3),
                    "killer_cutoff_rate": round(
                        stats.killer_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    "tt_move_cutoff_rate": round(
                        stats.tt_move_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    **_table_row(stats, spec),
                    "value": stats.value,
                    "seconds": round(seconds, 3),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2}  {spec.name:<22} {stats.nodes:>10,} nodes  "
                f"{row['gain']:>5}x  hit {row['tt_hit_rate']:>6.1%}  "
                f"ret {row['tt_return_rate']:>6.1%}  entries {stats.tt_entries:>9,}  "
                f"{seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def transposition_by_game(depth: int, games: int, seed: int, opening_plies: int) -> list[dict]:
    """What a whole game costs, and the only view in which persistence means anything.

    A table that survives between moves is measured here and nowhere else, because
    "between moves" does not exist in a sweep over independent positions. It is measured
    rather than adopted: carrying state from one search into the next is exactly what
    makes a search stop being a function of the position it is given.
    """
    rng = random.Random(seed)
    openings = [[rng.randrange(7) for _ in range(opening_plies)] for _ in range(games)]
    rows = []
    baseline = baseline_seconds = None
    for spec in TRANSPOSITIONS + (PERSISTENT,):
        agent = NegamaxAgent(
            depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=spec,
            scouting=NO_SCOUT, seed=seed,
        )
        total: dict = {}
        seconds = 0.0
        plies = 0
        played = []
        for game, opening in enumerate(openings):
            agent.new_game(seed + game)
            board = Bitboard()
            for col in opening:
                if board.is_terminal() or not board.can_play(col):
                    break
                board.play(col)
            while not board.is_terminal():
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                _accumulate(total, stats)
                plies += 1
                board.play(move)
            played.append("".join(str(col + 1) for col in board.history))
        nodes, cutoffs = total["nodes"], total["cutoffs"]
        if baseline is None:
            baseline, baseline_seconds, baseline_games = nodes, seconds, played
        rows.append(
            {
                "depth": depth,
                "evaluation": DEFAULT.name,
                "ordering": DYNAMIC.name,
                "transposition": spec.name,
                "games": games,
                "opening_plies": opening_plies,
                "moves": plies,
                "same_games": played == baseline_games,
                "nodes": nodes,
                "nodes_per_move": round(nodes / plies, 1),
                "cutoffs": cutoffs,
                "gain": round(baseline / nodes, 3),
                "time_gain": round(baseline_seconds / seconds, 3),
                "first_cutoff_rate": round(total["first_cutoffs"] / cutoffs, 4),
                "killer_cutoff_rate": round(total["killer_cutoffs"] / cutoffs, 4),
                "tt_move_cutoff_rate": round(total["tt_move_cutoffs"] / cutoffs, 4),
                **_table_row(_Totals(total), spec),
                "seconds": round(seconds, 2),
                "seconds_per_move": round(seconds / plies, 4),
            }
        )
        row = rows[-1]
        print(
            f"{spec.name:<22} {nodes:>11,} nodes over {plies} moves  {row['gain']:>5}x  "
            f"hit {row['tt_hit_rate']:>6.1%}  same games {str(row['same_games']):<5}  "
            f"{seconds:7.2f}s",
            flush=True,
        )
    return rows


def _deepening_row(stats, spec: Deepening) -> dict:
    """The counters iterative deepening adds, kept apart from the node count for the same
    reason Phase 7 kept the ordering quality apart from it: how deep a search got and how
    much it spent are different questions, and under a clock the first one is the result."""
    return {
        "iterations": stats.iterations,
        "depth_reached": stats.depth,
        "aborted": stats.aborted,
    }


def deepening_by_position(depths: tuple[int, ...]) -> list[dict]:
    """The gate of Phase 9 over the 200 frozen reference positions.

    Every value is checked against the stored one as it goes, so the sweep is the
    measurement and the correctness check at once -- the same arrangement Phase 8 used,
    and it has more to prove here. Iterative deepening is the first optimisation in this
    project that is allowed to change the *move*, so the value contract is the only thing
    left holding it: an iteration that completed at depth d must return exactly the value
    a direct depth-d search returns, and `changed_move` counts, rather than hides, the
    cases where a different column of equal value came back.

    `even` answers from depth d-1 when d is odd, so it is checked against the reference
    record for the depth it actually answered from. Comparing it against depth d would be
    comparing it against a question it deliberately declined to answer.
    """
    data = reference.load()
    values = {(r["moves"], r["depth"]): r["value"] for r in data["records"]}
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        baseline = baseline_seconds = None
        baseline_moves: list = []
        for spec in DEEPENINGS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                deepening=spec, scouting=NO_SCOUT, seed=2026,
            )
            answer_depth = depth if spec.answers(depth) else depth - 1
            total: dict = {}
            seconds = 0.0
            chosen, reached, aborted = [], [], 0
            for record in records:
                board = Bitboard.from_move_string(record["moves"])
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                assert stats.value == values[(record["moves"], answer_depth)], (
                    f"{spec.name} at depth {depth} changed the value of {record['moves']!r}"
                )
                assert stats.depth == answer_depth, f"{spec.name} answered from {stats.depth}"
                _accumulate(total, stats)
                chosen.append(move)
                reached.append(stats.depth)
                aborted += stats.aborted
            nodes, cutoffs = total["nodes"], total["cutoffs"]
            if baseline is None:
                baseline, baseline_seconds, baseline_moves = nodes, seconds, chosen
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": TABLE.name,
                    "deepening": spec.name,
                    "budget": BUDGET,
                    "positions": len(records),
                    "nodes": nodes,
                    "leaves": total["leaves"],
                    "cutoffs": cutoffs,
                    "gain": round(baseline / nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "ebf": round((nodes / len(records)) ** (1 / depth), 4),
                    "changed_move": sum(a != b for a, b in zip(chosen, baseline_moves)),
                    "answer_depth": answer_depth,
                    "aborted": aborted,
                    "first_cutoff_rate": round(total["first_cutoffs"] / cutoffs, 4),
                    "mean_cut_index": round(total["cut_index_total"] / cutoffs, 3),
                    "killer_cutoff_rate": round(total["killer_cutoffs"] / cutoffs, 4),
                    "tt_move_cutoff_rate": round(total["tt_move_cutoffs"] / cutoffs, 4),
                    **_table_row(_Totals(total), TABLE),
                    "seconds": round(seconds, 2),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth}  {spec.name:<26} {nodes:>10,} nodes  {row['gain']:>5}x  "
                f"first-cut {row['first_cutoff_rate']:>6.1%}  "
                f"killer-cut {row['killer_cutoff_rate']:>6.1%}  "
                f"tt-cut {row['tt_move_cutoff_rate']:>6.1%}  "
                f"moved differently {row['changed_move']:>3}/{len(records)}  {seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def deepening_on_empty_board(depths: tuple[int, ...]) -> list[dict]:
    """The one position where the surcharge can be stated as an identity.

    With the transposition table switched off, `naive` carries nothing at all between
    iterations, so its node count must be *exactly* the sum of the separate searches at
    depths 1..d. That is asserted here rather than assumed, which turns the textbook
    "about b/(b-1) extra work" into a number that can be wrong.
    """
    rows = []
    for depth in depths:
        # The identity, computed the slow honest way: d separate searches, added up. The
        # table has to be off on BOTH sides of this ratio or it is not the surcharge of
        # iterating -- it is the surcharge of iterating minus what the shared table gives
        # back, which is a different quantity and belongs in the `gain` column.
        separate = sum(
            NegamaxAgent(
                k, ordering=DYNAMIC, evaluation=DEFAULT, transposition=NO_TABLE,
                scouting=NO_SCOUT, seed=2026,
            ).choose_move(Bitboard())[1].nodes
            for k in range(1, depth + 1)
        )
        direct_untabled = NegamaxAgent(
            depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=NO_TABLE,
            scouting=NO_SCOUT, seed=2026,
        ).choose_move(Bitboard())[1].nodes
        untabled = NegamaxAgent(
            depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=NO_TABLE,
            deepening=NAIVE, scouting=NO_SCOUT, seed=2026,
        ).choose_move(Bitboard())[1]
        assert untabled.nodes == separate, (
            f"naive/no_table at depth {depth}: {untabled.nodes} != {separate}"
        )

        baseline = baseline_seconds = None
        baseline_move = None
        for spec in DEEPENINGS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                deepening=spec, scouting=NO_SCOUT, seed=2026,
            )
            started = time.perf_counter()
            move, stats = agent.choose_move(Bitboard())
            seconds = time.perf_counter() - started
            if baseline is None:
                baseline, baseline_seconds, baseline_move = stats.nodes, seconds, move
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": TABLE.name,
                    "deepening": spec.name,
                    "budget": BUDGET,
                    "nodes": stats.nodes,
                    "leaves": stats.leaves,
                    "cutoffs": stats.cutoffs,
                    "gain": round(baseline / stats.nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "separate_searches": separate,
                    "direct_no_table": direct_untabled,
                    "surcharge_no_table": round(separate / direct_untabled, 3),
                    "changed_move": int(move != baseline_move),
                    "first_cutoff_rate": round(stats.first_cutoff_rate, 4),
                    "killer_cutoff_rate": round(
                        stats.killer_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    "tt_move_cutoff_rate": round(
                        stats.tt_move_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    **_table_row(stats, TABLE),
                    **_deepening_row(stats, spec),
                    "value": stats.value,
                    "seconds": round(seconds, 3),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2}  {spec.name:<26} {stats.nodes:>10,} nodes  "
                f"{row['gain']:>5}x  tt-cut {row['tt_move_cutoff_rate']:>6.1%}  "
                f"killer-cut {row['killer_cutoff_rate']:>6.1%}  "
                f"answered from {stats.depth:>2}  {seconds:7.2f}s",
                flush=True,
            )
        print(f"           separate searches 1..{depth}, no table: {separate:,} "
              f"= {separate / direct_untabled:.3f}x the untabled direct search "
              f"({direct_untabled:,})\n")
    return rows


def deepening_by_game(depth: int, games: int, seed: int, opening_plies: int) -> list[dict]:
    """The third view, as in Phases 7 and 8: whole games rather than isolated positions.

    This is also where the move change becomes visible as something other than a counter.
    Every phase so far has ended with "all settings played exactly the same moves"; this
    one cannot, and `same_games` is the field that says so.
    """
    rng = random.Random(seed)
    openings = [[rng.randrange(7) for _ in range(opening_plies)] for _ in range(games)]
    rows = []
    baseline = baseline_seconds = None
    baseline_games: list = []
    for spec in DEEPENINGS:
        agent = NegamaxAgent(
            depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
            deepening=spec, scouting=NO_SCOUT, seed=seed,
        )
        total: dict = {}
        seconds = 0.0
        plies = 0
        played = []
        for game, opening in enumerate(openings):
            agent.new_game(seed + game)
            board = Bitboard()
            for col in opening:
                if board.is_terminal() or not board.can_play(col):
                    break
                board.play(col)
            while not board.is_terminal():
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                _accumulate(total, stats)
                plies += 1
                board.play(move)
            played.append("".join(str(col + 1) for col in board.history))
        nodes, cutoffs = total["nodes"], total["cutoffs"]
        if baseline is None:
            baseline, baseline_seconds, baseline_games = nodes, seconds, played
        rows.append(
            {
                "depth": depth,
                "evaluation": DEFAULT.name,
                "ordering": DYNAMIC.name,
                "transposition": TABLE.name,
                "deepening": spec.name,
                "budget": BUDGET,
                "games": games,
                "opening_plies": opening_plies,
                "moves": plies,
                "same_games": played == baseline_games,
                "games_that_differ": sum(a != b for a, b in zip(played, baseline_games)),
                "nodes": nodes,
                "nodes_per_move": round(nodes / plies, 1),
                "cutoffs": cutoffs,
                "gain": round(baseline / nodes, 3),
                "time_gain": round(baseline_seconds / seconds, 3),
                "first_cutoff_rate": round(total["first_cutoffs"] / cutoffs, 4),
                "killer_cutoff_rate": round(total["killer_cutoffs"] / cutoffs, 4),
                "tt_move_cutoff_rate": round(total["tt_move_cutoffs"] / cutoffs, 4),
                **_table_row(_Totals(total), TABLE),
                "seconds": round(seconds, 2),
                "seconds_per_move": round(seconds / plies, 4),
            }
        )
        row = rows[-1]
        print(
            f"{spec.name:<26} {nodes:>11,} nodes over {plies} moves  {row['gain']:>5}x  "
            f"tt-cut {row['tt_move_cutoff_rate']:>6.1%}  "
            f"games differing {row['games_that_differ']}/{games}  {seconds:7.2f}s",
            flush=True,
        )
    return rows


def deepening_under_a_clock(budgets: tuple[float, ...], positions: int) -> list[dict]:
    """The only measurement in this project that a clock is allowed to produce.

    Everything else is budgeted by depth or by nodes precisely so that it reproduces. What
    is being measured here *is* the clock: whether the budget is honoured, and how deep a
    search gets inside it. The overshoot is reported rather than asserted -- an assertion
    would turn a loaded machine into a failing test, and the number itself is the result.

    The depth column is the point of the whole phase. It is not a setting any more.
    """
    sample = [Bitboard()] + [
        Bitboard.from_move_string(moves)
        for moves in reference.sample_positions(positions, seed=2026)[1:]
    ]
    rows = []
    for budget in budgets:
        for spec in (ITERATIVE, EVEN):
            agent = NegamaxAgent(
                42, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                deepening=spec, scouting=NO_SCOUT, seed=2026,
            )
            over, depths, nodes, aborted = [], [], 0, 0
            for board in sample:
                _, stats = agent.choose_move(board, Budget(time=budget))
                over.append(stats.elapsed - budget)
                depths.append(stats.depth)
                nodes += stats.nodes
                aborted += stats.aborted
            rows.append(
                {
                    "budget_seconds": budget,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": TABLE.name,
                    "deepening": spec.name,
                    "budget": "time",
                    "positions": len(sample),
                    "worst_overshoot_ms": round(max(over) * 1000, 3),
                    "worst_overshoot_share": round(max(over) / budget, 5),
                    "mean_overshoot_ms": round(sum(over) / len(over) * 1000, 3),
                    "exceeded": sum(value > 0 for value in over),
                    "min_depth": min(depths),
                    "median_depth": sorted(depths)[len(depths) // 2],
                    "max_depth": max(depths),
                    "aborted": aborted,
                    "nodes": nodes,
                }
            )
            row = rows[-1]
            print(
                f"budget {budget:>6}s  {spec.name:<26} worst overshoot "
                f"{row['worst_overshoot_ms']:>+8.2f} ms ({row['worst_overshoot_share']:>+7.2%})"
                f"  depth {row['min_depth']:>2}-{row['max_depth']:>2} "
                f"(median {row['median_depth']:>2})  aborted {aborted}/{len(sample)}",
                flush=True,
            )
    return rows


# ------------------------------------------------------------------ Phase 10, negascout


def _scout_row(stats_or_total, spec: Scout) -> dict:
    """The counters NegaScout adds. `research_rate` is not a diagnostic: it is the
    quantity the algorithm bets on, so it is reported beside every node count it
    produced rather than inferred from the saving afterwards."""
    scouts = stats_or_total.scout_searches
    return {
        "scout_searches": scouts,
        "researches": stats_or_total.researches,
        "research_rate": round(stats_or_total.researches / scouts, 4) if scouts else "",
    }


def _rate(part: int, whole: int, places: int = 4) -> float:
    """A depth-1 alpha-beta search records no cutoffs at all -- the root never cuts -- so
    every cutoff-derived rate has to survive an empty denominator."""
    return round(part / whole, places) if whole else 0.0


def scout_by_position(depths: tuple[int, ...]) -> list[dict]:
    """The gate of Phase 10 over the 200 frozen reference positions.

    Every value is checked against the stored one as it goes, and so is every *move*:
    unlike Phase 9, NegaScout is back in the category of Phases 7 and 8 and provably
    cannot name a different one of several equally good moves, so `changed_move` is not a
    counter to be interpreted -- it has to be zero, and the sweep says so rather than
    assuming it.

    Four settings. `no_scout` is Phase 9's direct search and every gain is quoted against
    it; `negascout` is the phase; `negascout+wide` throws the scout's bound away and
    re-searches from alpha instead, which is the control for the classic bug of this
    algorithm; `negascout+from3` leaves the degenerate shallow depths alone, where a null
    window has nothing to cut and a re-search is pure loss.
    """
    data = reference.load()
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        baseline = baseline_seconds = None
        baseline_moves: list = []
        for spec in SCOUTS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                scouting=spec, seed=2026,
            )
            total: dict = {}
            seconds = 0.0
            chosen = []
            for record in records:
                board = Bitboard.from_move_string(record["moves"])
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                assert stats.value == record["value"], (
                    f"{spec.name} at depth {depth} changed the value of {record['moves']!r}"
                )
                _accumulate(total, stats)
                chosen.append(move)
            nodes, cutoffs = total["nodes"], total["cutoffs"]
            if baseline is None:
                baseline, baseline_seconds, baseline_moves = nodes, seconds, chosen
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": TABLE.name,
                    "deepening": DIRECT.name,
                    "scouting": spec.name,
                    "budget": BUDGET,
                    "positions": len(records),
                    "nodes": nodes,
                    "leaves": total["leaves"],
                    "cutoffs": cutoffs,
                    "gain": round(baseline / nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "ebf": round((nodes / len(records)) ** (1 / depth), 4),
                    "changed_move": sum(a != b for a, b in zip(chosen, baseline_moves)),
                    "first_cutoff_rate": _rate(total["first_cutoffs"], cutoffs),
                    "mean_cut_index": _rate(total["cut_index_total"], cutoffs, 3),
                    "killer_cutoff_rate": _rate(total["killer_cutoffs"], cutoffs),
                    "tt_move_cutoff_rate": _rate(total["tt_move_cutoffs"], cutoffs),
                    **_table_row(_Totals(total), TABLE),
                    **_scout_row(_Totals(total), spec),
                    "seconds": round(seconds, 2),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth}  {spec.name:<18} {nodes:>10,} nodes  {row['gain']:>5}x  "
                f"time {row['time_gain']:>5}x  "
                f"re-search {row['research_rate'] or 0:>7.2%} of {row['scout_searches']:>7,}  "
                f"tt ret {row['tt_return_rate']:>6.1%}  tt-cut {row['tt_move_cutoff_rate']:>6.1%}"
                f"  moved differently {row['changed_move']:>3}/{len(records)}  {seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def scout_on_empty_board(depths: tuple[int, ...]) -> list[dict]:
    """The same four settings on the one position everybody benchmarks.

    Reported for the reason Phases 7 and 8 report it: the empty board is alpha-beta's
    best case (thesis point P11) and it has gone both ways before -- move ordering lost
    here, the transposition table won here. It is also the only view in which the
    degenerate depths are visible one row at a time, and they are where this phase loses.
    """
    rows = []
    for depth in depths:
        baseline = baseline_seconds = baseline_value = None
        baseline_move = None
        for spec in SCOUTS:
            agent = NegamaxAgent(
                depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                scouting=spec, seed=2026,
            )
            started = time.perf_counter()
            move, stats = agent.choose_move(Bitboard())
            seconds = time.perf_counter() - started
            if baseline is None:
                baseline, baseline_seconds = stats.nodes, seconds
                baseline_value, baseline_move = stats.value, move
            assert stats.value == baseline_value, f"{spec.name} changed the value at {depth}"
            rows.append(
                {
                    "depth": depth,
                    "evaluation": DEFAULT.name,
                    "ordering": DYNAMIC.name,
                    "transposition": TABLE.name,
                    "deepening": DIRECT.name,
                    "scouting": spec.name,
                    "budget": BUDGET,
                    "nodes": stats.nodes,
                    "leaves": stats.leaves,
                    "cutoffs": stats.cutoffs,
                    "gain": round(baseline / stats.nodes, 3),
                    "time_gain": round(baseline_seconds / seconds, 3),
                    "changed_move": int(move != baseline_move),
                    "first_cutoff_rate": round(stats.first_cutoff_rate, 4),
                    "killer_cutoff_rate": round(
                        stats.killer_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    "tt_move_cutoff_rate": round(
                        stats.tt_move_cutoffs / stats.cutoffs if stats.cutoffs else 0.0, 4
                    ),
                    **_table_row(stats, TABLE),
                    **_scout_row(stats, spec),
                    "value": stats.value,
                    "seconds": round(seconds, 3),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth:>2}  {spec.name:<18} {stats.nodes:>10,} nodes  "
                f"{row['gain']:>5}x  time {row['time_gain']:>5}x  "
                f"re-search {row['research_rate'] or 0:>7.2%} of {row['scout_searches']:>6,}  "
                f"tt ret {row['tt_return_rate']:>6.1%}  {seconds:7.2f}s",
                flush=True,
            )
        print()
    return rows


def scout_by_game(depth: int, games: int, seed: int, opening_plies: int) -> list[dict]:
    """The third view, as in Phases 7, 8 and 9: complete games rather than positions.

    A game walks from an opening, where there is a large tree to prune, into an endgame
    where there is hardly any, so the average over a game is a different number from the
    average over a position set and both belong in the results. `same_games` is the
    property Phase 9 broke and this phase restores: with the deepening switch off,
    NegaScout cannot change a move, so every setting must play the identical games.
    """
    rng = random.Random(seed)
    openings = [[rng.randrange(7) for _ in range(opening_plies)] for _ in range(games)]
    rows = []
    baseline = baseline_seconds = None
    baseline_games: list = []
    for spec in SCOUTS:
        agent = NegamaxAgent(
            depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
            scouting=spec, seed=seed,
        )
        total: dict = {}
        seconds = 0.0
        plies = 0
        played = []
        for game, opening in enumerate(openings):
            agent.new_game(seed + game)
            board = Bitboard()
            for col in opening:
                if not board.is_terminal() and board.can_play(col):
                    board.play(col)
            while not board.is_terminal():
                started = time.perf_counter()
                move, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                _accumulate(total, stats)
                plies += 1
                board.play(move)
            played.append("".join(str(col + 1) for col in board.history))
        nodes, cutoffs = total["nodes"], total["cutoffs"]
        if baseline is None:
            baseline, baseline_seconds, baseline_games = nodes, seconds, played
        rows.append(
            {
                "depth": depth,
                "evaluation": DEFAULT.name,
                "ordering": DYNAMIC.name,
                "transposition": TABLE.name,
                "deepening": DIRECT.name,
                "scouting": spec.name,
                "budget": BUDGET,
                "games": games,
                "opening_plies": opening_plies,
                "moves": plies,
                "same_games": played == baseline_games,
                "games_that_differ": sum(a != b for a, b in zip(played, baseline_games)),
                "nodes": nodes,
                "nodes_per_move": round(nodes / plies, 1),
                "cutoffs": cutoffs,
                "gain": round(baseline / nodes, 3),
                "time_gain": round(baseline_seconds / seconds, 3),
                "first_cutoff_rate": round(total["first_cutoffs"] / cutoffs, 4),
                "killer_cutoff_rate": round(total["killer_cutoffs"] / cutoffs, 4),
                "tt_move_cutoff_rate": round(total["tt_move_cutoffs"] / cutoffs, 4),
                **_table_row(_Totals(total), TABLE),
                **_scout_row(_Totals(total), spec),
                "seconds": round(seconds, 2),
                "seconds_per_move": round(seconds / plies, 4),
            }
        )
        row = rows[-1]
        print(
            f"{spec.name:<18} {nodes:>11,} nodes over {plies} moves  {row['gain']:>5}x  "
            f"re-search {row['research_rate'] or 0:>7.2%}  "
            f"same games {str(row['same_games']):<5}  {seconds:7.2f}s",
            flush=True,
        )
    return rows


def scout_by_ordering(depths: tuple[int, ...]) -> list[dict]:
    """Slide 148's warning, measured on this phase the way Phase 7 measured it on its own.

    NegaScout is supposed to win when the first move tried is usually the best one and to
    lose when it is not, because every wrong guess costs a whole re-search. The four base
    orders are the same four Phase 7 used, so the row for `center_first+killers+history`
    is directly comparable with the position sweep and the two failure modes are the ones
    the slide names.
    """
    data = reference.load()
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        for ordering in (DYNAMIC, STATIC, LEFT, RANDOM):
            baseline = baseline_seconds = None
            for spec in (NO_SCOUT, SCOUT):
                agent = NegamaxAgent(
                    depth, ordering=ordering, evaluation=DEFAULT, transposition=TABLE,
                    scouting=spec, seed=2026,
                )
                total: dict = {}
                seconds = 0.0
                for record in records:
                    board = Bitboard.from_move_string(record["moves"])
                    started = time.perf_counter()
                    _, stats = agent.choose_move(board)
                    seconds += time.perf_counter() - started
                    assert stats.value == record["value"], (
                        f"{ordering.name}/{spec.name} changed the value of "
                        f"{record['moves']!r} at depth {depth}"
                    )
                    _accumulate(total, stats)
                nodes, cutoffs = total["nodes"], total["cutoffs"]
                if baseline is None:
                    baseline, baseline_seconds = nodes, seconds
                rows.append(
                    {
                        "depth": depth,
                        "evaluation": DEFAULT.name,
                        "ordering": ordering.name,
                        "transposition": TABLE.name,
                        "deepening": DIRECT.name,
                        "scouting": spec.name,
                        "budget": BUDGET,
                        "positions": len(records),
                        "nodes": nodes,
                        "cutoffs": cutoffs,
                        "gain": round(baseline / nodes, 3),
                        "time_gain": round(baseline_seconds / seconds, 3),
                        "first_cutoff_rate": _rate(total["first_cutoffs"], cutoffs),
                        **_scout_row(_Totals(total), spec),
                        "seconds": round(seconds, 2),
                    }
                )
                row = rows[-1]
                print(
                    f"depth {depth}  {ordering.name:<30} {spec.name:<10} "
                    f"{nodes:>10,} nodes  {row['gain']:>5}x  "
                    f"first-cut {row['first_cutoff_rate']:>6.1%}  "
                    f"re-search {row['research_rate'] or 0:>7.2%}  {seconds:7.2f}s",
                    flush=True,
                )
        print()
    return rows


def scout_under_deepening(depths: tuple[int, ...]) -> list[dict]:
    """The two phases that were supposed to help each other, measured together.

    Phase 9 found that under iterative deepening the transposition move causes 24.6-37.5%
    of cutoffs against 0.2-0.6% in a direct search, i.e. the first move tried is a far
    better guess. That is exactly the condition NegaScout needs, so the gain should be
    *larger* here than it is over a direct search. Both gains are quoted against their own
    baseline, which is the only way the comparison means anything.

    Aspiration windows at the root -- narrowing the *root* window around the previous
    iteration's value -- are the natural next step from here and are deliberately out of
    scope: they can fail low or high at the root, which needs a widening loop, and a
    widening loop is a place where a value can be lost. Named here rather than half done.
    """
    data = reference.load()
    rows = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        for deepening in (DIRECT, ITERATIVE):
            baseline = baseline_seconds = None
            for spec in (NO_SCOUT, SCOUT):
                agent = NegamaxAgent(
                    depth, ordering=DYNAMIC, evaluation=DEFAULT, transposition=TABLE,
                    deepening=deepening, scouting=spec, seed=2026,
                )
                total: dict = {}
                seconds = 0.0
                for record in records:
                    board = Bitboard.from_move_string(record["moves"])
                    started = time.perf_counter()
                    _, stats = agent.choose_move(board)
                    seconds += time.perf_counter() - started
                    assert stats.value == record["value"], (
                        f"{deepening.name}/{spec.name} changed the value of "
                        f"{record['moves']!r} at depth {depth}"
                    )
                    _accumulate(total, stats)
                nodes, cutoffs = total["nodes"], total["cutoffs"]
                if baseline is None:
                    baseline, baseline_seconds = nodes, seconds
                rows.append(
                    {
                        "depth": depth,
                        "evaluation": DEFAULT.name,
                        "ordering": DYNAMIC.name,
                        "transposition": TABLE.name,
                        "deepening": deepening.name,
                        "scouting": spec.name,
                        "budget": BUDGET,
                        "positions": len(records),
                        "nodes": nodes,
                        "cutoffs": cutoffs,
                        "gain": round(baseline / nodes, 3),
                        "time_gain": round(baseline_seconds / seconds, 3),
                        "first_cutoff_rate": _rate(total["first_cutoffs"], cutoffs),
                        "killer_cutoff_rate": _rate(total["killer_cutoffs"], cutoffs),
                        "tt_move_cutoff_rate": _rate(total["tt_move_cutoffs"], cutoffs),
                        **_table_row(_Totals(total), TABLE),
                        **_scout_row(_Totals(total), spec),
                        "seconds": round(seconds, 2),
                    }
                )
                row = rows[-1]
                print(
                    f"depth {depth}  {deepening.name:<22} {spec.name:<10} "
                    f"{nodes:>10,} nodes  {row['gain']:>5}x  "
                    f"first-cut {row['first_cutoff_rate']:>6.1%}  "
                    f"tt-cut {row['tt_move_cutoff_rate']:>6.1%}  "
                    f"re-search {row['research_rate'] or 0:>7.2%}  {seconds:7.2f}s",
                    flush=True,
                )
        print()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", action="store_true")

    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--ordering", action="store_true")
    parser.add_argument("--ordering-games", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--table-empty", action="store_true")
    parser.add_argument("--table-games", action="store_true")
    parser.add_argument("--deepening", action="store_true")
    parser.add_argument("--deepening-empty", action="store_true")
    parser.add_argument("--deepening-games", action="store_true")
    parser.add_argument("--deepening-clock", action="store_true")
    parser.add_argument("--scout", action="store_true")
    parser.add_argument("--scout-empty", action="store_true")
    parser.add_argument("--scout-games", action="store_true")
    parser.add_argument("--scout-ordering", action="store_true")
    parser.add_argument("--scout-deepening", action="store_true")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--timed-depth", type=int, default=6)
    parser.add_argument("--unpruned-depth", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--opening-plies", type=int, default=4)
    args = parser.parse_args()

    if args.nodes:
        write(nodes_by_depth(args.max_depth, args.timed_depth), NODES_CSV)
    if args.curve:
        # The zero-evaluation rows are the control. Greedy is a two-ply search whose only
        # tie-break is centre order, which is what a two-ply minimax with no evaluation
        # also reduces to -- so that row measures the evaluation function on its own,
        # with depth held fixed, and the depth-4 pair does the same one level deeper.
        plan = [
            (1, 2000, DEFAULT),
            (2, 2000, DEFAULT),
            (3, 2000, DEFAULT),
            (4, 600, DEFAULT),
            (5, 200, DEFAULT),
            (2, 2000, ZERO),
            (4, 600, ZERO),
        ]
        write(score_vs_greedy(plan, args.seed, args.opening_plies), CURVE_CSV)
    if args.pruning:
        write(
            pruning_by_depth(args.max_depth, args.unpruned_depth, args.max_seconds),
            PRUNING_CSV,
        )
    if args.positions:
        write(pruning_by_position(args.max_depth), POSITIONS_CSV)
    if args.ordering:
        write(ordering_on_empty_board((4, 5, 6, 7, 8, 9, 10)), ORDERING_EMPTY_CSV)
        write(ordering_by_position((4, 5, 6, 7)), ORDERING_CSV)
    if args.ordering_games:
        write(
            ordering_by_game(6, games=10, seed=args.seed, opening_plies=args.opening_plies),
            ORDERING_GAMES_CSV,
        )
    if args.density:
        write(transposition_density((2, 3, 4, 5, 6, 7), positions=40), DENSITY_CSV)
    if args.table:
        write(transposition_by_position((4, 5, 6, 7)), TABLE_CSV)
    if args.table_empty:
        write(
            transposition_on_empty_board((4, 5, 6, 7, 8, 9, 10, 11, 12)),
            TABLE_EMPTY_CSV,
        )
    if args.table_games:
        write(
            transposition_by_game(
                6, games=10, seed=args.seed, opening_plies=args.opening_plies
            ),
            TABLE_GAMES_CSV,
        )
    if args.deepening:
        write(deepening_by_position((4, 5, 6, 7)), DEEPENING_CSV)
    if args.deepening_empty:
        write(deepening_on_empty_board((4, 5, 6, 7, 8, 9, 10, 11, 12)), DEEPENING_EMPTY_CSV)
    if args.deepening_games:
        write(
            deepening_by_game(
                6, games=10, seed=args.seed, opening_plies=args.opening_plies
            ),
            DEEPENING_GAMES_CSV,
        )
    if args.deepening_clock:
        write(
            deepening_under_a_clock((0.02, 0.05, 0.1, 0.25, 0.5, 1.0), positions=25),
            DEEPENING_CLOCK_CSV,
        )
    if args.scout:
        write(scout_by_position((1, 2, 3, 4, 5, 6, 7)), SCOUT_CSV)
    if args.scout_empty:
        write(
            scout_on_empty_board((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
            SCOUT_EMPTY_CSV,
        )
    if args.scout_games:
        write(
            scout_by_game(6, games=10, seed=args.seed, opening_plies=args.opening_plies),
            SCOUT_GAMES_CSV,
        )
    if args.scout_ordering:
        write(scout_by_ordering((5, 6, 7)), SCOUT_ORDERING_CSV)
    if args.scout_deepening:
        write(scout_under_deepening((4, 5, 6, 7)), SCOUT_DEEPENING_CSV)


if __name__ == "__main__":
    main()
