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

    python -m experiments.scaling --nodes
    python -m experiments.scaling --curve
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from agents.greedy_agent import GreedyAgent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from experiments.match import Tally, play_match
from search.evaluation import DEFAULT, ZERO

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
NODES_CSV = RESULTS / "minimax_nodes_by_depth.csv"
CURVE_CSV = RESULTS / "minimax_vs_greedy.csv"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", action="store_true")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--timed-depth", type=int, default=6)
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


if __name__ == "__main__":
    main()
