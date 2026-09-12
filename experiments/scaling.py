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

    python -m experiments.scaling --nodes
    python -m experiments.scaling --curve
    python -m experiments.scaling --pruning
    python -m experiments.scaling --positions
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from agents.alphabeta import NegamaxAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from experiments import reference
from experiments.match import Tally, play_match
from search.evaluation import DEFAULT, ZERO
from search.ordering import CENTER_FIRST

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
NODES_CSV = RESULTS / "minimax_nodes_by_depth.csv"
CURVE_CSV = RESULTS / "minimax_vs_greedy.csv"
PRUNING_CSV = RESULTS / "alphabeta_nodes_by_depth.csv"
POSITIONS_CSV = RESULTS / "alphabeta_by_position.csv"

# The ordering every node count in this phase was measured under. Phase 7 varies it, and
# compares against these numbers, so it is recorded in the output rather than assumed.
ORDERING = "center_first" if CENTER_FIRST == (3, 2, 4, 1, 5, 0, 6) else str(CENTER_FIRST)


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
            _, plain = NegamaxAgent(depth, pruning=False, evaluation=ZERO).choose_move(Bitboard())
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
            _, stats = NegamaxAgent(depth, pruning=True, evaluation=evaluation).choose_move(
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
        agent = NegamaxAgent(record["depth"], pruning=True, evaluation=DEFAULT)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", action="store_true")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--positions", action="store_true")
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


if __name__ == "__main__":
    main()
