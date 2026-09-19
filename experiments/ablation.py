"""Phase 14 -- the ablation matrix, measured as COMBINATIONS rather than one switch at a time.

**No course algorithm is introduced here.** Every algorithm this module measures was built,
gated and committed in Phases 5-10. What is new is the *shape* of the measurement.

Why combinations, and not a list of per-optimisation percentages (thesis point P33)
-----------------------------------------------------------------------------------
Every sweep committed before this phase moves ONE switch and holds the rest fixed, which
answers "what is this worth on top of what I already have" and not "what is this worth".
Those are different questions, and the difference is not a rounding error: measured over
the reference positions, move ordering and NegaScout together deliver only about four
fifths of the product of their separate gains.

The mechanism is Knuth-Moore. A search does not merely compute the root value, it *proves*
it, and the proof has a minimum size -- with a perfect ordering alpha-beta visits the
minimal tree. Every optimisation here removes a different waste *relative to that same
floor*: move ordering removes examining five useless children at a cut node before the
refuting one, the transposition table removes proving one position twice, NegaScout removes
computing an exact value where "better or not" was the whole question. Once the tree is
close to the proof there is very little left for the third one to skip, whichever third one
it is. So the gains overlap, and multiplying them out names a number the program never
reaches.

This module therefore measures all eight combinations of three switches and reports, beside
each measured gain, the PRODUCT of the separate gains -- so the overlap is visible as a
number instead of as an argument.

The population is named, every time (thesis point P11)
------------------------------------------------------
The empty board is alpha-beta's best case and minimax's worst case, so it overstates the
ratio from both ends. The matrix is measured over the frozen reference positions, using the
same sub-samples the committed sweeps use (200 positions at depths 4 and 5, 40 at depth 6,
10 at depth 7). That is deliberate: it makes two rows of this matrix directly comparable
with numbers already in `data/results/`, and those comparisons are asserted as the run goes
-- see `_CROSS_CHECKS`.

Nothing here imports pygame or matplotlib; `experiments/` stays authoritative and headless.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from agents.alphabeta import NegamaxAgent
from core.bitboard import Bitboard
from experiments import reference
from search.deepening import DIRECT
from search.evaluation import DEFAULT
from search.ordering import DYNAMIC, STATIC
from search.scouting import NO_SCOUT, SCOUT
from search.transposition import NO_TABLE, TABLE

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
MATRIX_CSV = RESULTS / "ablation_matrix.csv"
CONTROL_CSV = RESULTS / "parallel_control.csv"

# The three switches. Each is (letter, name, off-spec, on-spec). The letter is what makes a
# combination readable in a table: "O-S" is ordering and NegaScout with no table.
SWITCHES: tuple[tuple[str, str, object, object], ...] = (
    ("O", "ordering", STATIC, DYNAMIC),
    ("T", "transposition", NO_TABLE, TABLE),
    ("S", "scouting", NO_SCOUT, SCOUT),
)

# The depths, and the sub-sample of the frozen reference set available at each. The set
# holds 200 positions at depths 1-5, 40 at depth 6 and 10 at depth 7, so the number of
# positions CHANGES with depth and every figure drawn from this file must normalise by it.
DEPTHS: tuple[int, ...] = (4, 5, 6, 7)

# Rows of already-committed CSVs that two of the eight combinations must reproduce exactly.
# `(combination, csv, column, {depth: value})`. These are not decoration: if the matrix
# disagrees with a committed file, the matrix is wrong and the run stops.
_CROSS_CHECKS: tuple[tuple[str, str, dict[int, int]], ...] = (
    # Phase 6 alpha-beta: static centre-first, no table, no scout.
    ("---", "alphabeta_by_position.csv", {4: 61727, 5: 227162, 6: 126483, 7: 99419}),
    # Phase 10's full stack: the same search the default agent runs.
    ("OTS", "scout_by_position.csv", {4: 41340, 5: 107645, 6: 46979, 7: 19463}),
)

BUDGET = "depth"


def combinations() -> list[tuple[str, dict]]:
    """The eight combinations, in a fixed order: none, then singles, pairs, the triple.

    Ordered by how many switches are on so the table reads as a ladder. Within one size the
    order is the order of `SWITCHES`, which keeps it stable across runs -- a figure that
    reorders itself between runs is not reproducible.
    """
    rows = []
    for size in range(len(SWITCHES) + 1):
        for on in itertools.combinations(range(len(SWITCHES)), size):
            key = "".join(
                letter if index in on else "-"
                for index, (letter, _, _, _) in enumerate(SWITCHES)
            )
            specs = {
                name: (spec_on if index in on else spec_off)
                for index, (_, name, spec_off, spec_on) in enumerate(SWITCHES)
            }
            rows.append((key, specs))
    return rows


def _accumulate(total: dict, stats) -> None:
    """Sum the counters this matrix reports over a sweep of searches.

    Deliberately a local copy of the six lines `experiments.scaling` uses rather than an
    import of it: that module is two thousand lines and this one is imported by a spawned
    worker, which would then pay for all of it at every process start.
    """
    for field in (
        "nodes", "leaves", "cutoffs", "first_cutoffs", "killer_cutoffs",
        "tt_move_cutoffs", "scout_searches", "researches",
    ):
        total[field] = total.get(field, 0) + getattr(stats, field)


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def matrix(depths: tuple[int, ...] = DEPTHS) -> list[dict]:
    """Every combination of the three switches, over the frozen reference positions.

    Each value is checked against the stored one as the sweep goes, so this run is at the
    same time the measurement and a correctness check: a combination that changed an answer
    would stop the sweep rather than quietly produce a better-looking number.
    """
    data = reference.load()
    rows: list[dict] = []
    for depth in depths:
        records = [record for record in data["records"] if record["depth"] == depth]
        baseline = baseline_seconds = None
        for key, specs in combinations():
            agent = NegamaxAgent(
                depth,
                evaluation=DEFAULT,
                ordering=specs["ordering"],
                transposition=specs["transposition"],
                scouting=specs["scouting"],
                deepening=DIRECT,
                seed=2026,
            )
            total: dict = {}
            seconds = 0.0
            for record in records:
                board = Bitboard.from_move_string(record["moves"])
                started = time.perf_counter()
                _, stats = agent.choose_move(board)
                seconds += time.perf_counter() - started
                assert stats.value == record["value"], (
                    f"combination {key} at depth {depth} changed the value of "
                    f"{record['moves']!r}"
                )
                _accumulate(total, stats)
            nodes, cutoffs = total["nodes"], total["cutoffs"]
            if baseline is None:
                baseline, baseline_seconds = nodes, seconds
            rows.append(
                {
                    "depth": depth,
                    "positions": len(records),
                    "combination": key,
                    "switches_on": len(SWITCHES) - key.count("-"),
                    "evaluation": DEFAULT.name,
                    "ordering": specs["ordering"].name,
                    "transposition": specs["transposition"].name,
                    "scouting": specs["scouting"].name,
                    "deepening": DIRECT.name,
                    "budget": BUDGET,
                    "nodes": nodes,
                    "leaves": total["leaves"],
                    "cutoffs": cutoffs,
                    "gain": round(baseline / nodes, 4),
                    "time_gain": round(baseline_seconds / seconds, 4),
                    "ebf": round((nodes / len(records)) ** (1 / depth), 4),
                    "first_cutoff_rate": _rate(total["first_cutoffs"], cutoffs),
                    "killer_cutoff_rate": _rate(total["killer_cutoffs"], cutoffs),
                    "tt_move_cutoff_rate": _rate(total["tt_move_cutoffs"], cutoffs),
                    "research_rate": _rate(total["researches"], total["scout_searches"]),
                    "seconds": round(seconds, 2),
                }
            )
            row = rows[-1]
            print(
                f"depth {depth}  n={len(records):>3}  {key}  "
                f"{nodes:>10,} nodes  {row['gain']:>7.3f}x  ebf {row['ebf']:>6.3f}  "
                f"first-cut {row['first_cutoff_rate']:>6.1%}  {seconds:7.2f}s",
                flush=True,
            )
        print(flush=True)
    return add_overlap(rows)


def add_overlap(rows: list[dict]) -> list[dict]:
    """Add the two columns the whole phase exists for: the PRODUCT and what was DELIVERED.

    For a combination of two or more switches, `product` is what the gain would be if the
    switches attacked independent waste -- the product of each one's gain measured on its
    own at the same depth -- and `delivered` is the measured gain as a fraction of it.
    Anything below 1.0 is overlap: work one optimisation saved that another would have
    saved anyway.

    A pure function over rows, separately from the measurement, so it can be tested and
    mutated without running a search.
    """
    singles: dict[tuple[int, str], float] = {}
    for row in rows:
        key = row["combination"]
        if len(SWITCHES) - key.count("-") == 1:
            singles[(row["depth"], key.replace("-", ""))] = row["gain"]
    for row in rows:
        key = row["combination"]
        letters = [letter for letter in key if letter != "-"]
        if len(letters) < 2:
            row["product"] = ""
            row["delivered"] = ""
            continue
        product = 1.0
        for letter in letters:
            product *= singles[(row["depth"], letter)]
        row["product"] = round(product, 4)
        row["delivered"] = round(row["gain"] / product, 4)
    return rows


def check_against_committed(rows: list[dict]) -> list[str]:
    """Two combinations must reproduce numbers already in `data/results/`, to the node.

    The baseline combination is Phase 6's alpha-beta and the full stack is Phase 10's, so
    if this matrix is measuring what it claims to measure those two columns are not new
    numbers at all. Returns the disagreements; an empty list is the check passing.
    """
    found = {(row["depth"], row["combination"]): row["nodes"] for row in rows}
    problems = []
    for key, source, expected in _CROSS_CHECKS:
        for depth, nodes in expected.items():
            measured = found.get((depth, key))
            if measured is None:
                continue
            if measured != nodes:
                problems.append(
                    f"combination {key} at depth {depth}: matrix says {measured:,} nodes, "
                    f"{source} says {nodes:,}"
                )
    return problems


# --------------------------------------------------------------------------------------
# The parallel control (thesis point P35)
# --------------------------------------------------------------------------------------
# Phase 12 measured the tournament harness at 1.86x / 2.67x / 3.32x / 3.58x on eight cores
# and argued that the missing speedup is the machine and not the harness, by comparing it
# with a program that cannot be inefficient. That control was run from a scratch script and
# its numbers were never committed to a CSV, so this phase re-measures it and commits it --
# a figure may not contain a number that is not in a committed file.


def _refuse_if_main_is_not_importable() -> None:
    """Refuse to build a spawn pool that would respawn dying workers for ever.

    `spawn` starts every child as a fresh interpreter which re-imports `__main__` to find
    the work function. Run from a heredoc (`python - <<EOF`), `__main__.__file__` is
    `<stdin>`, so the child is told to load a path that does not exist, dies with
    FileNotFoundError -- and `Pool` repairs it by spawning a replacement, for ever.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if path is None or not os.path.isfile(path):
        raise RuntimeError(
            "the parallel control needs an importable __main__ (its workers are spawned, "
            f"and each re-imports it); got {path!r}. Run it as "
            "`python -m experiments.ablation --control`, never from a heredoc."
        )


def _burn(iterations: int) -> int:
    """A task that cannot be inefficient: no allocation, no I/O, one integer each way.

    Whatever this measures is the machine. It is deliberately arithmetic in the interpreter
    rather than anything numeric or vectorised, because that is what the tournament workers
    are doing too.
    """
    total = 0
    for index in range(iterations):
        total += index * index % 7
    return total


def cpu_control(
    worker_counts: tuple[int, ...] = (1, 2, 4, 6, 8),
    tasks: int = 32,
    iterations: int = 4_000_000,
    repeats: int = 5,
) -> list[dict]:
    """The same shape as the tournament sweep: identical tasks, perfectly balanced.

    `tasks` is a multiple of every worker count in the default sweep, so load imbalance is
    not in the answer -- which is the point. Phase 12 already measured separately that
    granularity is worth 9% and the process boundary 0.196 s.

    **Repeated, because one sample of this is not a result.** A speedup measured once on a
    fanless laptop that is also running an editor is reproducible to about +-4% at two
    workers and only +-23% at eight; the serial baseline alone drifted 15.1 s to 18.6 s
    across five runs. Every sample is recorded and the figure draws the spread, for the
    same reason every rate in this project carries a sigma.

    MUST be started as `python -m experiments.ablation --control`. See
    `_refuse_if_main_is_not_importable`.
    """
    rows = []
    for sample in range(1, repeats + 1):
        baseline = None
        for workers in worker_counts:
            if workers > 1:
                _refuse_if_main_is_not_importable()
            started = time.perf_counter()
            if workers == 1:
                results = [_burn(iterations) for _ in range(tasks)]
            else:
                with mp.get_context("spawn").Pool(workers) as pool:
                    results = pool.map(_burn, [iterations] * tasks)
            seconds = time.perf_counter() - started
            assert len(set(results)) == 1, "the control tasks are not identical"
            if baseline is None:
                baseline = seconds
            rows.append(
                {
                    "config": "cpu_control",
                    "variable": "workers",
                    "sample": sample,
                    "tasks": tasks,
                    "iterations_per_task": iterations,
                    "workers": workers,
                    "seconds": round(seconds, 2),
                    "tasks_per_second": round(tasks / seconds, 3),
                    "speedup": round(baseline / seconds, 3),
                    "efficiency": round(baseline / seconds / workers, 3),
                }
            )
            row = rows[-1]
            print(
                f"sample {sample}  {workers} worker(s): {seconds:7.2f}s  "
                f"{row['speedup']:>6.3f}x  efficiency {row['efficiency']:.1%}",
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
    parser.add_argument("--matrix", action="store_true", help="the 8-combination ablation")
    parser.add_argument("--control", action="store_true", help="the synthetic CPU control")
    parser.add_argument("--depths", type=str, default="", help="e.g. 4,5,6,7")
    parser.add_argument("--iterations", type=int, default=4_000_000)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-write", action="store_true",
                        help="measure and print, but leave the committed CSV alone")
    args = parser.parse_args()

    if args.matrix:
        depths = (
            tuple(int(part) for part in args.depths.split(","))
            if args.depths
            else DEPTHS
        )
        rows = matrix(depths)
        problems = check_against_committed(rows)
        for problem in problems:
            print(f"DISAGREES WITH A COMMITTED CSV: {problem}", flush=True)
        assert not problems, "the matrix disagrees with an already-committed measurement"
        print("cross-checks against committed CSVs: all agree", flush=True)
        if not args.no_write:
            write(rows, MATRIX_CSV)

    if args.control:
        rows = cpu_control(tasks=args.tasks, iterations=args.iterations,
                           repeats=args.repeats)
        if not args.no_write:
            write(rows, CONTROL_CSV)


if __name__ == "__main__":
    main()
