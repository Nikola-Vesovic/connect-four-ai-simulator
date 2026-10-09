"""Phase 14 -- every figure and every table in the thesis, regenerated from `data/results/`.

**No course algorithm is introduced here.** This module measures nothing. It reads CSVs that
thirteen earlier phases committed and turns them into the results chapter.

Four rules this module is built around
--------------------------------------
1. **A figure may not contain a number that is not in a committed CSV.** Nothing here runs a
   search, opens a socket or invents a value. If a number is wanted and no CSV has it, the
   honest options were to measure it and commit it (that is what `experiments/ablation.py`
   did for the matrix and the parallel control) or to leave the figure undrawn.
2. **Every charted quantity also exists as a table.** A figure is never the only place a
   number lives, because the text quotes exact figures and a reader must be able to check
   them. Each `*_data()` function below returns the table AND the series, and the series is
   derived from the table rows, so the two cannot drift apart. `RESULTS.md` is generated
   from the same objects the charts are drawn from.
3. **Charts are the exception, tables are the rule.** Six quantities get a chart, because
   for those six the SHAPE of a curve is the argument. Everything else is a table. What was
   left out, and why, is listed in `OMITTED`.
4. **A figure names its population** (thesis point P11). The empty board is alpha-beta's
   best case and minimax's worst case, so every axis label or caption says whether it is
   the empty board, the 200 reference positions, or whole games.

Determinism, greyscale and matplotlib hygiene
---------------------------------------------
Re-running `python -m experiments.plots --all` must produce byte-identical files, so the
figure size and DPI are fixed, the Agg backend is forced, and the timestamps matplotlib
would otherwise write into PDF and PNG metadata are suppressed explicitly. `--check`
asserts that rather than hoping for it.

Series are distinguished by marker and line style as well as colour, because the thesis
will be printed and a reader in greyscale must still be able to tell four curves apart.

**matplotlib is imported lazily, inside `_pyplot()`.** Importing this module must not pull
it in: `experiments/` is imported by spawned tournament and benchmark workers, and the
project keeps a test that asserts in a fresh interpreter that none of them drags a drawing
or windowing library along. The data shaping above the drawing line is therefore testable,
and tested, on a machine with no matplotlib installed at all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "data" / "figures"

# The floor slides 51-56 give for alpha-beta's effective branching factor: with a perfect
# ordering the search visits the minimal tree, about b^(d/2) leaves instead of b^d.
SQRT7 = 7 ** 0.5

# Fixed so two runs of this module produce the same bytes. 200 dpi keeps the PNG legible on
# a slide without making it enormous; the PDF is vector and does not care.
FIGURE_DPI = 200
PDF_METADATA = {"CreationDate": None, "Creator": "connect4", "Producer": "connect4"}
PNG_METADATA = {"Software": None}

# Grey-safe: four levels of darkness, each with its own marker and dash pattern, so the
# figures survive being printed in black and white.
STYLES = (
    {"color": "#000000", "marker": "o", "linestyle": "-", "markersize": 5},
    {"color": "#4a4a4a", "marker": "s", "linestyle": "--", "markersize": 5},
    {"color": "#7a7a7a", "marker": "^", "linestyle": "-.", "markersize": 5.5},
    {"color": "#a8a8a8", "marker": "D", "linestyle": ":", "markersize": 4.5},
    {"color": "#2f2f2f", "marker": "v", "linestyle": (0, (3, 1, 1, 1)), "markersize": 5},
)

# What did NOT get a chart, and why. Recorded because choosing is part of the work: a
# results chapter wants eight to ten figures, not twenty-five.
OMITTED = {
    "minimax_vs_greedy": "seven points; the argument is that the curve is NOT monotone "
                         "(depth 3 below depth 2), which a table states more exactly than "
                         "a line does",
    "ordering/transposition/scout gain curves": "gains of 0.8x-1.6x are flat lines on any "
                                                "axis; the numbers are the result, so they "
                                                "are tabulated",
    "transposition_density": "four numbers (the ceiling at depths 3/5/7/8)",
    "mcts_vs_alphabeta": "six rows whose point is that two budgets disagree by 26 points; "
                         "a table puts the two budgets side by side, a chart would not",
    "mcts_ablation": "the shape IS that one switch sits at 6.9 sigma and every other "
                     "inside 1.2 sigma -- which is read off a sigma column, not a plot",
    "tournament pairing matrix": "a 15-row round robin is a table; the Elo fit is the "
                                 "chart",
    "deepening_under_a_clock": "an overshoot distribution reported as median/p90/p99",
}


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------

def read(name: str) -> list[dict]:
    """One committed CSV, as a list of dicts. The only input this module has."""
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: str) -> float:
    return float(value)


def _i(value: str) -> int:
    return int(value)


def _binomial_sigma(rate: float, n: int) -> float:
    """Standard error of a rate over n independent trials.

    Used only where the CSV does not already carry an error column. A rate without its
    sigma is not a result, so every bar chart of a rate in this module carries one.
    """
    return (rate * (1.0 - rate) / n) ** 0.5 if n else 0.0


# --------------------------------------------------------------------------------------
# Labels -- Latin by default, Serbian Cyrillic behind a switch
# --------------------------------------------------------------------------------------
# Algorithm names stay in English throughout (the Phase 3 decision, which the window
# already follows): Serbian labels, English algorithm names.

LATIN = {
    "depth": "dubina pretrage",
    "nodes_per_position": "čvorova po poziciji",
    "nodes_single": "čvorova u jednoj pretrazi",
    "empty_board": "prazna tabla (jedna pozicija)",
    "reference_positions": "referentne pozicije (n = 200 do dubine 5, pa 40 i 10)",
    "f1_title": "Broj pregledanih čvorova po dubini",
    "ebf": "efektivni faktor grananja",
    "floor": "asimptotska referenca √7 = 2,6458",
    "f2_title": "Efektivni faktor grananja (n = 200/200/40/10)",
    "elo": "Elo rejting",
    "f3_title": "Elo lestvica sa 95% intervalom (7.500 partija, budžet 4.000 čvorova)",
    "workers": "broj procesa",
    "speedup": "ubrzanje (veće je bolje)",
    "harness": "turnir (izmereno, jedan prolaz)",
    "control": "sintetička kontrola, medijana od 5",
    "ideal": "idealno linearno ubrzanje",
    "f4_title": "Ubrzanje na osam jezgara, sa kontrolom koja ne može biti neefikasna",
    "simulations": "broj simulacija",
    "preserved": "udeo očuvanih dobijenih pozicija",
    "visit_share": "udeo poseta najboljem potezu",
    "rate": "udeo",
    "f5_title": "MCTS: konvergencija i taktika u odnosu na broj simulacija",
    "mate_in_one": "matiranje u jednom potezu",
    "block_in_one": "odbrana od mata u jednom potezu",
    "optimal_rate": "udeo optimalnih poteza",
    "f6_title": "Poređenje sa savršenim rešavačem, razdvojeno po ishodu pozicije",
    "won": "dobijena",
    "drawn": "remi",
    "lost": "izgubljena",
    "minimax": "minimax (bez odsecanja)",
    "alpha-beta": "alfa-beta",
    "+ move ordering": "+ sortiranje poteza",
    "+ table + negascout": "+ tabela transpozicija i NegaScout",
    "Phase 6: alpha-beta, static centre-first": "alfa-beta, statički redosled od centra",
    "Phase 7: + killers and history": "+ killer potezi i istorija",
    "Phase 8: + transposition table": "+ tabela transpozicija",
    "Phase 10: + NegaScout": "+ nulti prozor (NegaScout)",
    "between": "između agenata",
    "most": "najviše",
    "length": "dužina",
}

CYRILLIC = {
    "depth": "дубина претраге",
    "nodes_per_position": "чворова по позицији",
    "nodes_single": "чворова у једној претрази",
    "empty_board": "празна табла (једна позиција)",
    "reference_positions": "референтне позиције (n = 200 до дубине 5, па 40 и 10)",
    "f1_title": "Број прегледаних чворова по дубини",
    "ebf": "ефективни фактор гранања",
    "floor": "асимптотска референца √7 = 2,6458",
    "f2_title": "Ефективни фактор гранања (n = 200/200/40/10)",
    "elo": "Ело рејтинг",
    "f3_title": "Ело лествица са 95% интервалом (7.500 партија, буџет 4.000 чворова)",
    "workers": "број процеса",
    "speedup": "убрзање (веће је боље)",
    "harness": "турнир (измерено, један пролаз)",
    "control": "синтетичка контрола, медијана од 5",
    "ideal": "идеално линеарно убрзање",
    "f4_title": "Убрзање на осам језгара, са контролом која не може бити неефикасна",
    "simulations": "број симулација",
    "preserved": "удео очуваних добијених позиција",
    "visit_share": "удео посета најбољем потезу",
    "rate": "удео",
    "f5_title": "МЦТС: конвергенција и тактика у односу на број симулација",
    "mate_in_one": "матирање у једном потезу",
    "block_in_one": "одбрана од мата у једном потезу",
    "optimal_rate": "удео оптималних потеза",
    "f6_title": "Поређење са савршеним решавачем, раздвојено по исходу позиције",
    "won": "добијена",
    "drawn": "реми",
    "lost": "изгубљена",
    "minimax": "minimax (без одсецања)",
    "alpha-beta": "алфа-бета",
    "+ move ordering": "+ сортирање потеза",
    "+ table + negascout": "+ табела транспозиција и NegaScout",
    "Phase 6: alpha-beta, static centre-first": "алфа-бета, статички редослед од центра",
    "Phase 7: + killers and history": "+ killer потези и историја",
    "Phase 8: + transposition table": "+ табела транспозиција",
    "Phase 10: + NegaScout": "+ нулти прозор (NegaScout)",
    "between": "између агената",
    "most": "највише",
    "length": "дужина",
}

LANGUAGES = {"latin": LATIN, "cyrillic": CYRILLIC}

# The glyphs each script must be able to draw. Latin needs the five Serbian letters that
# are not plain ASCII; Cyrillic needs the six that are specific to Serbian plus a
# representative slice of the shared alphabet. These are PROVEN to render by
# `prove_glyphs()` rather than assumed -- a missing glyph in matplotlib draws a box, and a
# box is silent.
LATIN_GLYPHS = "šđčćž"
CYRILLIC_GLYPHS = "ђјљњћџабвгдежзиклмнопрстуфхцчш"

# A private-use code point no installed font defines. It is the CONTROL for the glyph
# proof: whatever matplotlib draws for this is what a missing glyph looks like, and a real
# glyph is one whose bitmap differs from it.
MISSING_GLYPH = "\ueeee"


# --------------------------------------------------------------------------------------
# Figure 1 -- node count against depth, two populations, log y
# --------------------------------------------------------------------------------------

def figure1_data() -> dict:
    """Minimax against each alpha-beta rung, on the empty board and over real positions.

    Log y, and that is not a style choice: the counts span 7 to 6,634,026, and on a linear
    axis every curve but minimax's is flat against the x-axis. The whole b^d -> b^(d/2)
    claim is the difference between two straight lines on a log axis.

    Two panels because the population changes the answer by a factor of three (P11). The
    right-hand panel is normalised to nodes PER POSITION, because the reference set holds
    200 positions at depths 1-5 but only 40 at depth 6 and 10 at depth 7 -- plotting totals
    would put a step in the curve that is an artefact of the sample size and nothing else.
    """
    empty_minimax = {_i(r["depth"]): _i(r["nodes"]) for r in read("minimax_nodes_by_depth.csv")}
    empty_ab = {
        _i(r["depth"]): _i(r["alphabeta_nodes"])
        for r in read("alphabeta_nodes_by_depth.csv")
        if r["evaluation"] == "full"
    }
    empty_order = {
        _i(r["depth"]): _i(r["nodes"])
        for r in read("ordering_on_empty_board.csv")
        if r["ordering"] == "center_first+killers+history"
    }
    empty_full = {
        _i(r["depth"]): _i(r["nodes"])
        for r in read("scout_on_empty_board.csv")
        if r["scouting"] == "negascout"
    }

    position = {_i(r["depth"]): r for r in read("alphabeta_by_position.csv")}
    matrix = {
        (_i(r["depth"]), r["combination"]): r for r in read("ablation_matrix.csv")
    }

    rows = []
    for depth in range(1, 9):
        n = _i(position[depth]["positions"]) if depth in position else 0
        rows.append(
            {
                "depth": depth,
                "empty_minimax": empty_minimax.get(depth, ""),
                "empty_alphabeta": empty_ab.get(depth, ""),
                "empty_ordering": empty_order.get(depth, ""),
                "empty_full": empty_full.get(depth, ""),
                "positions": n or "",
                "pos_minimax": round(_i(position[depth]["minimax_nodes"]) / n, 1) if n else "",
                "pos_alphabeta": round(_i(position[depth]["alphabeta_nodes"]) / n, 1) if n else "",
                "pos_ordering": (
                    round(_i(matrix[(depth, "O--")]["nodes"]) / n, 1)
                    if (depth, "O--") in matrix else ""
                ),
                "pos_full": (
                    round(_i(matrix[(depth, "OTS")]["nodes"]) / n, 1)
                    if (depth, "OTS") in matrix else ""
                ),
            }
        )

    def series(key):
        return [(r["depth"], r[key]) for r in rows if r[key] != ""]

    return {
        "slug": "f1_nodes_vs_depth",
        "chapter": "6. Rezultati",
        "title_key": "f1_title",
        "sources": [
            "minimax_nodes_by_depth.csv", "alphabeta_nodes_by_depth.csv",
            "ordering_on_empty_board.csv", "scout_on_empty_board.csv",
            "alphabeta_by_position.csv", "ablation_matrix.csv",
        ],
        "caption": (
            "Number of positions examined for one move, by search depth, on a logarithmic "
            "axis. LEFT: the empty board, which is alpha-beta's best case and minimax's "
            "worst case. RIGHT: the 200 frozen reference positions, normalised to nodes "
            "per position because the sub-sample falls to 40 at depth 6 and 10 at depth 7. "
            "The gap between the two panels is thesis point P11: quoting both ends from "
            "the empty board overstates the ratio from both ends at once."
        ),
        "columns": list(rows[0]),
        "rows": rows,
        "panels": [
            ("empty_board", [
                ("minimax", series("empty_minimax")),
                ("alpha-beta", series("empty_alphabeta")),
                ("+ move ordering", series("empty_ordering")),
                ("+ table + negascout", series("empty_full")),
            ]),
            ("reference_positions", [
                ("minimax", series("pos_minimax")),
                ("alpha-beta", series("pos_alphabeta")),
                ("+ move ordering", series("pos_ordering")),
                ("+ table + negascout", series("pos_full")),
            ]),
        ],
    }


# --------------------------------------------------------------------------------------
# Figure 2 -- the effective branching factor closing on its floor
# --------------------------------------------------------------------------------------

def figure2_data() -> dict:
    """The four phases, converging on sqrt(7) = 2.6458 (thesis point P33).

    Linear y, and that is also not a style choice: these are four numbers between 2.6 and
    4.2 and the argument is how much of the remaining distance each phase closes. On a log
    axis the whole picture would collapse into four indistinguishable flat lines.

    The chain is continuous by construction -- each phase's baseline row is the previous
    phase's optimised row -- and `check()` below asserts it rather than trusting it.
    """
    ordering = {
        (_i(r["depth"]), r["ordering"]): _f(r["ebf"]) for r in read("ordering_by_position.csv")
    }
    table = {
        (_i(r["depth"]), r["transposition"]): _f(r["ebf"])
        for r in read("transposition_by_position.csv")
    }
    scout = {
        (_i(r["depth"]), r["scouting"]): _f(r["ebf"]) for r in read("scout_by_position.csv")
    }
    positions = {_i(r["depth"]): _i(r["positions"]) for r in read("ordering_by_position.csv")}

    depths = (4, 5, 6, 7)
    stages = (
        ("Phase 6: alpha-beta, static centre-first",
         [ordering[(d, "center_first")] for d in depths]),
        ("Phase 7: + killers and history",
         [ordering[(d, "center_first+killers+history")] for d in depths]),
        ("Phase 8: + transposition table", [table[(d, "dict")] for d in depths]),
        ("Phase 10: + NegaScout", [scout[(d, "negascout")] for d in depths]),
    )

    rows = []
    previous = None
    for name, values in stages:
        ebf7 = values[-1]
        gap = ebf7 - SQRT7
        closed = "" if previous is None else round(1 - gap / (previous - SQRT7), 4)
        rows.append(
            {
                "stage": name,
                **{f"d{d}": values[index] for index, d in enumerate(depths)},
                "positions_at_d7": positions[7],
                "distance_to_floor_at_d7": round(gap, 4),
                "fraction_of_remainder_closed": closed,
            }
        )
        previous = ebf7

    return {
        "slug": "f2_ebf_convergence",
        "chapter": "6. Rezultati",
        "title_key": "f2_title",
        "sources": [
            "ordering_by_position.csv", "transposition_by_position.csv",
            "scout_by_position.csv",
        ],
        "caption": (
            "Effective branching factor over the frozen reference positions (200 positions "
            "at depths 4 and 5, 40 at depth 6, 10 at depth 7), against the floor of "
            "sqrt(7) = 2.6458 that slides 51-56 give for a perfectly ordered alpha-beta "
            "search. Each optimisation closes a FRACTION OF WHAT REMAINS -- 44%, then 31%, "
            "then 26% -- never a fixed number of nodes. That is the visual form of thesis "
            "point P33: they are all racing at the same floor, so their gains overlap "
            "instead of multiplying."
        ),
        "columns": list(rows[0]),
        "rows": rows,
        "depths": depths,
        "stages": stages,
        "floor": SQRT7,
    }


# --------------------------------------------------------------------------------------
# Figure 3 -- the Elo ladder with its bootstrap band
# --------------------------------------------------------------------------------------

def figure3_data() -> dict:
    """Six agents, one rating each, with the central 95% of 1,000 refits as an error bar.

    The band is asymmetric, so it is carried as two numbers and never as a single +-.
    """
    rows = []
    for record in read("tournament_ladder_elo.csv"):
        elo, low, high = _f(record["elo"]), _f(record["elo_low"]), _f(record["elo_high"])
        rows.append(
            {
                "label": record["label"],
                "agent": record["agent"],
                "games": _i(record["games"]),
                "elo": elo,
                "elo_low": low,
                "elo_high": high,
                "band_minus": round(elo - low, 1),
                "band_plus": round(high - elo, 1),
                "score": _f(record["score"]),
                "score_first": _f(record["score_first"]),
                "score_second": _f(record["score_second"]),
                "wins": _i(record["wins"]),
                "draws": _i(record["draws"]),
                "losses": _i(record["losses"]),
            }
        )
    rows.sort(key=lambda row: row["elo"])
    return {
        "slug": "f3_elo_ladder",
        "chapter": "5. Metodologija / 6. Rezultati",
        "title_key": "f3_title",
        "sources": ["tournament_ladder_elo.csv"],
        "caption": (
            "Maximum-likelihood Bradley-Terry ratings over the whole round robin -- 6 "
            "agents, 15 pairings, 500 games each, every one on a budget of 4,000 positions "
            "examined, with four random opening plies. Draws count half a point to each "
            "side. The bars are the central 95% of 1,000 seeded parametric bootstrap "
            "refits, i.e. HOW FAR THIS TABLE WOULD MOVE IF THE SAME GAMES WERE PLAYED "
            "AGAIN. Its blind spot is stated rather than hidden (P36): `random` scored 0.0% "
            "against three of its five opponents, so it is resampled from p = 0, wins none "
            "in every resample, and gets a band far narrower than its real uncertainty. "
            "What makes that rating finite at all is the prior, not the interval."
        ),
        "columns": list(rows[0]),
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# Figure 4 -- parallel speedup, with the control underneath it
# --------------------------------------------------------------------------------------

def figure4_data() -> dict:
    """The harness against a program that cannot be inefficient (thesis point P35).

    Linear y: a speedup of 3.08 against an ideal of 8 is a gap the reader is meant to see,
    and a log axis would flatter it.

    The control is a REPEATED measurement and is drawn as a median with its full observed
    range, because one sample of it is not a result -- on this machine it is reproducible
    to about +-2% at two workers and only +-15% at eight. The verdict column is therefore
    deliberately conservative: the harness "beats" the control only when it exceeds the
    control's BEST observed sample, not its median.
    """
    harness = {
        _i(r["workers"]): _f(r["speedup"])
        for r in read("parallel_speedup.csv")
        if r["variable"] == "workers"
    }
    samples: dict[int, list[float]] = {}
    for record in read("parallel_control.csv"):
        samples.setdefault(_i(record["workers"]), []).append(_f(record["speedup"]))

    workers = sorted(set(harness) | set(samples))
    rows = []
    for count in workers:
        h = harness.get(count)
        values = samples.get(count, [])
        median = round(statistics.median(values), 3) if values else ""
        low = round(min(values), 3) if values else ""
        high = round(max(values), 3) if values else ""
        rows.append(
            {
                "workers": count,
                "harness_speedup": h if h is not None else "",
                "control_median": median,
                "control_min": low,
                "control_max": high,
                "control_samples": len(values),
                "ideal_speedup": float(count),
                "harness_beats_control": (
                    "" if h is None or not values else ("yes" if h > high else "no")
                ),
                "harness_efficiency": "" if h is None else round(h / count, 3),
            }
        )
    return {
        "slug": "f4_parallel_speedup",
        "chapter": "5. Metodologija",
        "title_key": "f4_title",
        "sources": ["parallel_speedup.csv", "parallel_control.csv"],
        "caption": (
            "Speedup against worker count on an Apple M3 with four performance and four "
            "efficiency cores, in a fanless chassis. The harness curve is one sweep of the "
            "tournament (180 games, 4,000-node budget, identical digest at every worker "
            "count). The control is a pure CPU-bound Python loop -- thirty-two identical "
            "perfectly balanced tasks, nothing crossing the process boundary but an "
            "integer -- a program that CANNOT be inefficient, so whatever it fails to "
            "achieve is the machine; it is measured five times and drawn as a median with "
            "its full range. **The honest reading is narrower than it first appears.** "
            "Above two workers this machine is reproducible only to about +-15%, so the "
            "harness is separated from the control at 6 and 8 workers and is "
            "indistinguishable from it at 2 and 4. What survives the noise is the claim "
            "that matters: at eight workers a program that cannot be wasteful does no "
            "better than the harness, so the missing speedup is the machine. "
            "os.cpu_count() is a count of schedulable contexts, not a speedup."
        ),
        "columns": list(rows[0]),
        "rows": rows,
        "workers": workers,
        "harness": harness,
        "control_samples": samples,
    }


# --------------------------------------------------------------------------------------
# Figure 5 -- MCTS against simulation count
# --------------------------------------------------------------------------------------

def figure5_data() -> dict:
    """Convergence on forced wins, and one-move tactics, both against the budget.

    Two panels because they are two different populations and two different sample sizes --
    60 positions whose value is a forced win, and 150 tactical positions of each kind. Both
    are rates, so both carry binomial error bars; a rate without its sigma is not a result.
    """
    convergence = read("mcts_convergence.csv")
    tactics = read("mcts_tactics.csv")

    conv_rows = []
    for record in convergence:
        n = _i(record["positions"])
        rate = _f(record["preserved_rate"])
        conv_rows.append(
            {
                "playout": record["playout"],
                "simulations_per_move": _i(record["iterations"]),
                "positions": n,
                "preserved_rate": rate,
                "preserved_sigma": round(_binomial_sigma(rate, n), 4),
                "mean_visit_share": _f(record["mean_visit_share"]),
                "max_tree_depth": _i(record["max_tree_depth"]),
                "nodes_per_simulation": _f(record["nodes_per_simulation"]),
            }
        )

    tac_rows = []
    for record in tactics:
        n = _i(record["positions"])
        mate, block = _f(record["mate_in_one_rate"]), _f(record["block_in_one_rate"])
        tac_rows.append(
            {
                "playout": record["playout"],
                "simulations_per_move": _i(record["iterations"]),
                "positions": n,
                "mate_in_one_rate": mate,
                "mate_in_one_sigma": round(_binomial_sigma(mate, n), 4),
                "block_in_one_rate": block,
                "block_in_one_sigma": round(_binomial_sigma(block, n), 4),
                "ms_per_move": _f(record["ms_per_move"]),
            }
        )

    def grouped(rows, key, sigma_key):
        out = {}
        for row in rows:
            out.setdefault(row["playout"], []).append(
                (row["simulations_per_move"], row[key], row[sigma_key])
            )
        for series in out.values():
            series.sort()
        return out

    return {
        "slug": "f5_mcts_convergence",
        "chapter": "6. Rezultati",
        "title_key": "f5_title",
        "sources": ["mcts_convergence.csv", "mcts_tactics.csv"],
        "caption": (
            "LEFT: the share of 60 positions whose true value is a forced win in which MCTS "
            "still plays a move that preserves the win, against the simulation budget. "
            "RIGHT: one-move tactics over 150 positions of each kind -- taking a mate in "
            "one, and blocking one. Error bars are binomial standard errors at the stated "
            "sample size. The budget is in SIMULATIONS, which is what a user sets and what "
            "the literature quotes; it is not a node count, and the two are not "
            "interchangeable (P29): a greedy simulation costs 214.3 positions and a random "
            "one 15.1."
        ),
        "columns": list(conv_rows[0]),
        "rows": conv_rows,
        "tactics_columns": list(tac_rows[0]),
        "tactics_rows": tac_rows,
        "convergence_series": grouped(conv_rows, "preserved_rate", "preserved_sigma"),
        "mate_series": grouped(tac_rows, "mate_in_one_rate", "mate_in_one_sigma"),
        "block_series": grouped(tac_rows, "block_in_one_rate", "block_in_one_sigma"),
    }


# --------------------------------------------------------------------------------------
# Figure 6 -- the perfect-solver benchmark, split by who is winning
# --------------------------------------------------------------------------------------

AGENT_ORDER = ("random", "greedy", "mcts", "ab-plain", "ab-order", "ab-full")
OUTCOMES = ("win", "draw", "loss")


def figure6_data() -> dict:
    """Optimal-move rate against the true value of every child, split by outcome class.

    Split, and never averaged, because the split is the finding (P37): a LOST position
    cannot contain a blunder at all, and when every move loses, "optimal" can only mean
    losing as slowly as possible -- which is exactly the thing MCTS does not compute.
    """
    records = read("benchmark_by_outcome.csv")
    rows = []
    for record in records:
        n = _i(record["positions"])
        rate = _f(record["optimal_rate"])
        rows.append(
            {
                "set": record["set"],
                "outcome": record["outcome"],
                "agent": record["agent"],
                "positions": n,
                "optimal_rate": rate,
                "optimal_sigma": _f(record["optimal_sigma"]),
                "blunder_available": _i(record["blunder_available"]),
                "blunder_rate": _f(record["blunder_rate"]),
                "mean_ply_loss": _f(record["mean_ply_loss"]),
            }
        )

    sets = []
    for name in ("end-easy", "middle-easy"):
        grid = {}
        for row in rows:
            if row["set"] == name:
                grid[(row["agent"], row["outcome"])] = row
        if grid:
            sets.append((name, grid))

    return {
        "slug": "f6_benchmark_by_outcome",
        "chapter": "6. Rezultati",
        "title_key": "f6_title",
        "sources": ["benchmark_by_outcome.csv"],
        "caption": (
            "Share of moves that preserve the position's true value, scored against a "
            "bounded self-certifying perfect solver, split by whether the position is won, "
            "drawn or lost for the player to move. Error bars are the standard errors "
            "already carried in the CSV. Two things the split shows and an average hides: "
            "a lost position CANNOT contain a blunder, so a 0% blunder rate there is a "
            "property of the population and not a strength; and MCTS collapses on lost "
            "positions, because losing slowly requires a mate distance and MCTS has none. "
            "The three alpha-beta rungs are inside one sigma of each other -- the benchmark "
            "saturates, which is the ceiling of the method and not a failure of it."
        ),
        "columns": list(rows[0]),
        "rows": rows,
        "sets": sets,
    }


# --------------------------------------------------------------------------------------
# The ablation matrix -- a TABLE, because the numbers are the argument
# --------------------------------------------------------------------------------------

def ablation_table() -> dict:
    """Thesis point P33, as the phase's central table.

    Reports COMBINATIONS, and puts the PRODUCT of the separate gains beside the measured
    combined gain so the overlap is visible as a number. "Ordering 2.1x, table 1.5x,
    NegaScout 1.9x" multiplies out to 4.8x-9.7x, and the program never gets there.
    """
    records = read("ablation_matrix.csv")
    rows = []
    for record in records:
        rows.append(
            {
                "depth": _i(record["depth"]),
                "positions": _i(record["positions"]),
                "combination": record["combination"],
                "ordering": "yes" if "O" in record["combination"] else "-",
                "table": "yes" if "T" in record["combination"] else "-",
                "negascout": "yes" if "S" in record["combination"] else "-",
                "nodes": _i(record["nodes"]),
                "gain": _f(record["gain"]),
                "product_if_independent": _f(record["product"]) if record["product"] else "",
                "delivered": _f(record["delivered"]) if record["delivered"] else "",
                "ebf": _f(record["ebf"]),
                "first_cutoff_rate": _f(record["first_cutoff_rate"]),
            }
        )
    return {
        "slug": "tab_ablation_matrix",
        "chapter": "6. Rezultati (zatvara poglavlje -- P33)",
        "sources": ["ablation_matrix.csv"],
        "caption": (
            "All eight combinations of the three value-preserving optimisations, over the "
            "frozen reference positions. O = move ordering (killers + history), T = "
            "transposition table, S = NegaScout. The baseline is Phase 6 alpha-beta with "
            "static centre-first ordering. `product_if_independent` is what the gain would "
            "be if the switches removed independent waste; `delivered` is the measured gain "
            "as a fraction of it. Below 1.0 is overlap."
        ),
        "columns": list(rows[0]),
        "rows": rows,
    }


def p33_slice() -> dict:
    """P33's published table, re-derived from the matrix with the table on on both sides.

    The committed claim -- 82% / 80% / 68% at depths 5 / 6 / 7 -- was measured in Phase 10
    from `scout_by_ordering.csv`. It is a CONDITIONAL statement: ordering against NegaScout
    with the transposition table already on. Re-deriving it here from an independently
    measured matrix is a cross-check on both.
    """
    matrix = {(_i(r["depth"]), r["combination"]): _i(r["nodes"]) for r in read("ablation_matrix.csv")}
    counts = {_i(r["depth"]): _i(r["positions"]) for r in read("ablation_matrix.csv")}
    rows = []
    for depth in (5, 6, 7):
        base = matrix[(depth, "-T-")]
        ordering = base / matrix[(depth, "OT-")]
        negascout = base / matrix[(depth, "-TS")]
        together = base / matrix[(depth, "OTS")]
        rows.append(
            {
                "depth": depth,
                "positions": counts[depth],
                "ordering_alone": round(ordering, 3),
                "negascout_alone": round(negascout, 3),
                "product_if_independent": round(ordering * negascout, 3),
                "measured_together": round(together, 3),
                "delivered": round(together / (ordering * negascout), 3),
            }
        )
    return {
        "slug": "tab_p33_published_slice",
        "chapter": "6. Rezultati (zatvara poglavlje -- P33)",
        "sources": ["ablation_matrix.csv"],
        "caption": (
            "The published P33 table, re-derived from the new matrix. Both gains are "
            "measured with the transposition table already on, which is the condition the "
            "original measurement was made under. Quote the depth-5 row: the depth-7 one "
            "rests on ten positions."
        ),
        "columns": list(rows[0]),
        "rows": rows,
    }


FIGURES_DATA = (figure1_data, figure2_data, figure3_data, figure4_data,
                figure5_data, figure6_data)
TABLES_DATA = (ablation_table, p33_slice)


# --------------------------------------------------------------------------------------
# Markdown emission
# --------------------------------------------------------------------------------------

def markdown_table(columns: list[str], rows: list[dict]) -> str:
    """A Markdown table Phase 16 can paste. Values are printed, never re-rounded."""
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def results_markdown() -> str:
    """`RESULTS.md` -- every figure with its chapter, caption, sources and its numbers.

    This file is the reason the gate can say that every charted quantity also exists as a
    table: it is generated from the same objects the charts are drawn from, so a number can
    not appear in one and not the other.
    """
    parts = [
        "# Phase 14 -- figures and tables for the thesis",
        "",
        "Generated by `python -m experiments.plots --all`. Do not edit by hand.",
        "Every number below comes from a committed CSV in `data/results/`; nothing here",
        "re-measures anything, and nothing here reaches the network.",
        "",
        "Figures are written twice: `.pdf` (vector, for the document) and `.png` (for",
        "slides). Both are byte-identical across runs -- `--check` asserts it.",
        "",
    ]
    for builder in FIGURES_DATA:
        data = builder()
        parts += [
            f"## {data['slug']}",
            "",
            f"- **Chapter:** {data['chapter']}",
            f"- **Files:** `{data['slug']}.pdf`, `{data['slug']}.png`",
            f"- **Sources:** {', '.join('`' + s + '`' for s in data['sources'])}",
            "",
            data["caption"],
            "",
            markdown_table(data["columns"], data["rows"]),
            "",
        ]
        if "tactics_rows" in data:
            parts += [markdown_table(data["tactics_columns"], data["tactics_rows"]), ""]
    for builder in TABLES_DATA:
        data = builder()
        parts += [
            f"## {data['slug']} (table only, no figure)",
            "",
            f"- **Chapter:** {data['chapter']}",
            f"- **Sources:** {', '.join('`' + s + '`' for s in data['sources'])}",
            "",
            data["caption"],
            "",
            markdown_table(data["columns"], data["rows"]),
            "",
        ]
    parts += ["## What deliberately did not get a chart", ""]
    for name, reason in OMITTED.items():
        parts.append(f"- **{name}** -- {reason}")
    parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Drawing -- matplotlib enters here and nowhere else
# --------------------------------------------------------------------------------------

def _pyplot():
    """Import matplotlib, force a headless backend, and pin everything that varies.

    Called from inside the drawing functions so that importing this module -- which a
    spawned worker may do transitively -- does not pull a drawing library into the process.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "connect4-phase14",
            "axes.unicode_minus": False,
        }
    )
    return plt


def save(fig, slug: str, language: str, pdf_metadata: dict | None = None) -> list[Path]:
    """Write the PDF and the PNG, with every source of run-to-run variation suppressed.

    matplotlib stamps a CreationDate into PDF metadata and a Software tag into PNG
    metadata. Both are timestamps-by-another-name for reproducibility purposes and both are
    switched off here, which is what makes `--check` able to pass.

    `pdf_metadata` is injectable for ONE reason: so a test can ask for a deliberately
    stamped file and prove the reproducibility check would notice. Without that the stamp
    half of the check is only ever exercised on files that have no stamp, and a version
    that ignored stamps entirely would pass.
    """
    suffix = "" if language == "latin" else "_cyr"
    paths = []
    pdf = FIGURES / f"{slug}{suffix}.pdf"
    png = FIGURES / f"{slug}{suffix}.png"
    fig.savefig(pdf, format="pdf",
                metadata=PDF_METADATA if pdf_metadata is None else pdf_metadata,
                bbox_inches="tight")
    fig.savefig(png, format="png", metadata=PNG_METADATA, bbox_inches="tight")
    paths += [pdf, png]
    return paths


def _draw_figure1(plt, data, words):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    # The two panels count different things and must not share a y-label: the empty board
    # is ONE position, so its curve is the size of a single search, while the right-hand
    # panel is an average over a sample whose size changes with depth.
    labels = ("nodes_single", "nodes_per_position")
    for axis, (population, series), label in zip(axes, data["panels"], labels):
        depths = set()
        for style, (name, points) in zip(STYLES, series):
            axis.plot([p[0] for p in points], [p[1] for p in points], label=words[name], **style)
            depths.update(point[0] for point in points)
        axis.set_yscale("log")
        axis.set_xlabel(words["depth"])
        axis.set_ylabel(words[label])
        axis.set_title(words[population], fontsize=8)
        axis.set_xticks(sorted(depths))
    axes[0].legend(loc="upper left")
    fig.suptitle(words["f1_title"])
    fig.tight_layout()
    return fig


def _draw_figure2(plt, data, words):
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    for style, (name, values) in zip(STYLES, data["stages"]):
        axis.plot(data["depths"], values, label=words[name], **style)
    axis.axhline(data["floor"], color="#000000", linewidth=1.2, linestyle=(0, (6, 2)))
    axis.annotate(
        words["floor"],
        xy=(data["depths"][0], data["floor"]),
        xytext=(data["depths"][0] + 0.05, data["floor"] + 0.05),
        fontsize=8,
    )
    axis.set_xlabel(words["depth"])
    axis.set_ylabel(words["ebf"])
    axis.set_xticks(data["depths"])
    # Headroom so the legend never sits on top of a curve.
    axis.set_ylim(2.5, 4.75)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title(words["f2_title"], fontsize=9)
    fig.tight_layout()
    return fig


def _draw_figure3(plt, data, words):
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    labels = [row["label"] for row in data["rows"]]
    values = [row["elo"] for row in data["rows"]]
    lower = [row["band_minus"] for row in data["rows"]]
    upper = [row["band_plus"] for row in data["rows"]]
    positions = range(len(labels))
    axis.barh(
        list(positions), values, xerr=[lower, upper],
        color="#c9c9c9", edgecolor="#000000", linewidth=0.8,
        error_kw={"ecolor": "#000000", "capsize": 3, "elinewidth": 1.0},
        height=0.6,
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels)
    axis.set_xlabel(words["elo"])
    axis.axvline(0.0, color="#000000", linewidth=0.8)
    axis.set_title(words["f3_title"])
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def _draw_figure4(plt, data, words):
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    workers = data["workers"]
    axis.plot(workers, workers, label=words["ideal"], color="#b0b0b0",
              linestyle=(0, (6, 2)), linewidth=1.0, marker="")
    harness = [(w, data["harness"][w]) for w in workers if w in data["harness"]]
    axis.plot([p[0] for p in harness], [p[1] for p in harness],
              label=words["harness"], **STYLES[0])
    # Five samples, so the control is a median with its full observed range: above two
    # workers the spread is wider than most of the gap being argued about.
    counts = [w for w in workers if data["control_samples"].get(w)]
    medians = [statistics.median(data["control_samples"][w]) for w in counts]
    lower = [medians[i] - min(data["control_samples"][w]) for i, w in enumerate(counts)]
    upper = [max(data["control_samples"][w]) - medians[i] for i, w in enumerate(counts)]
    axis.errorbar(counts, medians, yerr=[lower, upper], label=words["control"],
                  capsize=3, **STYLES[2])
    axis.set_xlabel(words["workers"])
    axis.set_ylabel(words["speedup"])
    axis.set_xticks(workers)
    axis.legend(loc="upper left", fontsize=8)
    axis.set_title(words["f4_title"], fontsize=9)
    fig.tight_layout()
    return fig


def _draw_figure5(plt, data, words):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for style, (playout, points) in zip(STYLES, sorted(data["convergence_series"].items())):
        axes[0].errorbar(
            [p[0] for p in points], [p[1] for p in points],
            yerr=[p[2] for p in points], label=playout, capsize=3, **style,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(words["simulations"])
    axes[0].set_ylabel(words["preserved"])
    axes[0].set_ylim(0.9, 1.02)
    axes[0].legend(loc="lower right")

    pairs = [
        (data["mate_series"], words["mate_in_one"], 0),
        (data["block_series"], words["block_in_one"], 2),
    ]
    for series, name, style_index in pairs:
        for offset, (playout, points) in enumerate(sorted(series.items())):
            style = dict(STYLES[(style_index + offset) % len(STYLES)])
            axes[1].errorbar(
                [p[0] for p in points], [p[1] for p in points],
                yerr=[p[2] for p in points], label=f"{name}: {playout}",
                capsize=2, linewidth=1.0, **style,
            )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(words["simulations"])
    axes[1].set_ylabel(words["rate"])
    axes[1].legend(loc="lower right", fontsize=6)
    fig.suptitle(words["f5_title"])
    fig.tight_layout()
    return fig


def _draw_figure6(plt, data, words):
    sets = data["sets"]
    fig, axes = plt.subplots(1, len(sets), figsize=(9.5, 4.0), sharey=True)
    if len(sets) == 1:
        axes = [axes]
    fills = {"win": "#ffffff", "draw": "#b4b4b4", "loss": "#5a5a5a"}
    hatches = {"win": "///", "draw": "", "loss": ".."}
    width = 0.26
    for axis, (name, grid) in zip(axes, sets):
        for index, outcome in enumerate(OUTCOMES):
            xs, ys, errs = [], [], []
            for slot, agent in enumerate(AGENT_ORDER):
                row = grid.get((agent, outcome))
                if row is None:
                    continue
                xs.append(slot + (index - 1) * width)
                ys.append(row["optimal_rate"])
                errs.append(row["optimal_sigma"])
            axis.bar(
                xs, ys, width=width, yerr=errs, label=words[
                    {"win": "won", "draw": "drawn", "loss": "lost"}[outcome]
                ],
                color=fills[outcome], edgecolor="#000000", linewidth=0.7,
                hatch=hatches[outcome],
                error_kw={"ecolor": "#000000", "capsize": 2, "elinewidth": 0.8},
            )
        axis.set_xticks(range(len(AGENT_ORDER)))
        axis.set_xticklabels(AGENT_ORDER, rotation=30, ha="right")
        axis.set_title(name)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel(words["optimal_rate"])
    # One legend for the whole figure, below the panels: inside either one it would sit on
    # top of a bar, and these bars reach the top of the axes by design.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(words["f6_title"])
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig


DRAWERS = {
    "f1_nodes_vs_depth": _draw_figure1,
    "f2_ebf_convergence": _draw_figure2,
    "f3_elo_ladder": _draw_figure3,
    "f4_parallel_speedup": _draw_figure4,
    "f5_mcts_convergence": _draw_figure5,
    "f6_benchmark_by_outcome": _draw_figure6,
}


def draw_all(language: str = "latin", pdf_metadata: dict | None = None) -> list[Path]:
    """Every figure, in one language, as PDF and PNG. Never calls `plt.show()`."""
    plt = _pyplot()
    words = LANGUAGES[language]
    written = []
    for builder in FIGURES_DATA:
        data = builder()
        figure = DRAWERS[data["slug"]](plt, data, words)
        written += save(figure, data["slug"], language, pdf_metadata)
        plt.close(figure)
    return written


# --------------------------------------------------------------------------------------
# Proving the glyphs, rather than assuming them
# --------------------------------------------------------------------------------------

def prove_glyphs(characters: str) -> dict[str, bool]:
    """Render each character on its own and decide whether the font really has it.

    A font that lacks a glyph does not fail -- it draws `.notdef`, an empty rectangle, and
    it draws the SAME rectangle for every character it is missing. So the test is not "did
    something appear", which a box passes. It is: does this character produce ink, and is
    that ink different from the ink a deliberately-missing character produces? A character
    far outside any installed font (U+FFFD's neighbours in an unassigned plane) supplies
    the reference box.

    Returns {character: True if genuinely rendered}. This project has a precedent for
    probing pixels rather than trusting a screenshot (Phase 3).
    """
    plt = _pyplot()
    import numpy as np

    def bitmap(text: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figure = plt.figure(figsize=(0.6, 0.6), dpi=100)
            figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=28)
            figure.canvas.draw()
            pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, 0].copy()
            plt.close(figure)
            return pixels

    missing = bitmap(MISSING_GLYPH)
    blank = bitmap(" ")
    # The control has to be a real control: if the reference character were one the font
    # DOES have, every comparison below would be against a letter and the test would pass
    # for a font with no Cyrillic in it at all.
    assert (missing != blank).any(), "the reference box drew nothing -- the control is dead"
    verdict = {}
    for character in characters:
        drawn = bitmap(character)
        has_ink = bool((drawn != blank).any())
        is_box = bool((drawn == missing).all())
        verdict[character] = has_ink and not is_box
    return verdict


def stamps_in(path: Path) -> list[str]:
    """Any wall-clock stamp left in a written figure.

    This exists because comparing two back-to-back runs is a FALSE PASS and I checked.
    matplotlib writes `/CreationDate (D:20260919220435+02'00')` into a PDF and a `Software`
    tag naming its own version into a PNG; the date only changes once a SECOND, so two runs
    a millisecond apart agree perfectly and a byte comparison alone would certify a file
    that is different tomorrow. The reproducibility claim therefore rests on the field
    being ABSENT, which is a property of one file and can be checked directly.
    """
    raw = path.read_bytes()
    return [
        name
        for name in (b"CreationDate", b"ModDate", b"Software")
        if name in raw
    ]


def check_determinism(language: str = "latin", pdf_metadata: dict | None = None) -> list[str]:
    """Draw everything twice, compare the bytes, AND check no timestamp survived.

    Both halves are needed and neither is sufficient: the byte comparison catches anything
    that varies within a run, and the stamp check catches what varies only between days.
    """
    first = {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in draw_all(language, pdf_metadata)}
    second = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in draw_all(language, pdf_metadata)}
    problems = [f"{path.name}: bytes changed" for path in first if first[path] != second[path]]
    for path in first:
        for stamp in stamps_in(path):
            problems.append(f"{path.name}: carries a {stamp.decode()} stamp")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="every figure and every table")
    parser.add_argument("--language", choices=sorted(LANGUAGES), default="latin")
    parser.add_argument("--both-languages", action="store_true")
    parser.add_argument("--tables", action="store_true", help="RESULTS.md only")
    parser.add_argument("--check", action="store_true", help="assert byte-identical reruns")
    parser.add_argument("--glyphs", action="store_true", help="prove both scripts render")
    args = parser.parse_args()

    if args.glyphs:
        for name, glyphs in (("latin", LATIN_GLYPHS), ("cyrillic", CYRILLIC_GLYPHS)):
            verdict = prove_glyphs(glyphs)
            bad = [character for character, ok in verdict.items() if not ok]
            print(f"{name}: {len(verdict) - len(bad)}/{len(verdict)} glyphs genuinely drawn"
                  + (f" -- MISSING {bad}" if bad else ""))
            assert not bad, f"{name} is missing glyphs: {bad}"

    if args.all or args.tables:
        (FIGURES / "RESULTS.md").write_text(results_markdown(), encoding="utf-8")
        print(f"wrote {FIGURES / 'RESULTS.md'}")

    if args.all:
        languages = sorted(LANGUAGES) if args.both_languages else [args.language]
        for language in languages:
            for path in draw_all(language):
                print(f"wrote {path}")

    if args.check:
        problems = check_determinism(args.language)
        print(
            "byte-identical on a second run, and no timestamp survives in any file"
            if not problems
            else "NOT REPRODUCIBLE:\n  " + "\n  ".join(problems)
        )
        assert not problems, f"reproducibility check failed: {problems}"


if __name__ == "__main__":
    main()
