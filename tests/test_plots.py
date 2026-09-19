"""Phase 14 -- tests for the ablation matrix and for the figure/table shaping.

Two rules shape this file.

**The data shaping is tested; the drawing is not.** A wrong argument to `ax.plot` is not a
fault this project can detect, and pretending otherwise would inflate the mutation score.
What CAN be tested, and is, is everything above the drawing line: how a CSV is read, how the
matrix is combined, how a product of separate gains is compared with the measured combined
gain, and whether a figure's numbers agree with the committed files they claim to come from.

**Most of this file runs with no matplotlib installed.** `experiments.plots` imports it
lazily, so every shaping test below works on a bare machine; only the handful that actually
render use `importorskip`. That is deliberate -- `requirements.txt` still holds nothing but
pygame and pytest, so `python main.py` runs on a machine that has never heard of matplotlib.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import ablation, plots

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"


# ---------------------------------------------------------------- the matrix, as a shape


def test_the_matrix_is_every_combination_exactly_once():
    keys = [key for key, _ in ablation.combinations()]
    assert keys == ["---", "O--", "-T-", "--S", "OT-", "O-S", "-TS", "OTS"]
    assert len(set(keys)) == 2 ** len(ablation.SWITCHES)


def test_a_switch_that_is_off_is_the_phase_before_it():
    """Every "off" position must be the spec the previous phase measured, or the matrix is
    not an ablation of this project's own ladder."""
    by_key = dict(ablation.combinations())
    off = by_key["---"]
    assert off["ordering"].name == "center_first"
    assert off["transposition"].name == "no_table"
    assert off["scouting"].name == "no_scout"
    on = by_key["OTS"]
    assert on["ordering"].name == "center_first+killers+history"
    assert on["transposition"].name == "dict"
    assert on["scouting"].name == "negascout"


def test_accumulating_sums_across_searches_instead_of_keeping_the_last_one():
    """A sweep adds up two hundred searches; overwriting would report the last position.

    Unit-tested rather than inferred from the committed CSV, because a test that only reads
    the file cannot see how the file was produced -- and the mutation check found exactly
    that hole.
    """
    class _Stats:
        def __init__(self, value):
            for field in ("nodes", "leaves", "cutoffs", "first_cutoffs", "killer_cutoffs",
                          "tt_move_cutoffs", "scout_searches", "researches"):
                setattr(self, field, value)

    total = {}
    ablation._accumulate(total, _Stats(3))
    ablation._accumulate(total, _Stats(4))
    assert total["nodes"] == 7
    assert total["cutoffs"] == 7
    assert total["researches"] == 7


def test_the_committed_matrix_reproduces_two_already_published_measurements():
    """The baseline is Phase 6's alpha-beta and the full stack is Phase 10's.

    So two of the eight rows are not new numbers at all, and if they disagree with the
    files those phases committed then the matrix is measuring something else.
    """
    rows = _matrix_rows()
    assert ablation.check_against_committed(rows) == []


def test_the_cross_check_would_notice_a_wrong_matrix():
    """The check above is only worth having if it can fail. Perturb one node count."""
    rows = [dict(row) for row in _matrix_rows()]
    for row in rows:
        if row["depth"] == 7 and row["combination"] == "---":
            row["nodes"] += 1
    assert ablation.check_against_committed(rows) != []


def test_overlap_is_the_measured_gain_over_the_product_of_the_separate_gains():
    """The one piece of arithmetic the whole phase exists for (thesis point P33)."""
    rows = [
        {"depth": 5, "combination": "---", "gain": 1.0},
        {"depth": 5, "combination": "O--", "gain": 2.0},
        {"depth": 5, "combination": "-T-", "gain": 3.0},
        {"depth": 5, "combination": "--S", "gain": 5.0},
        {"depth": 5, "combination": "OT-", "gain": 3.0},
        {"depth": 5, "combination": "O-S", "gain": 10.0},
        {"depth": 5, "combination": "-TS", "gain": 12.0},
        {"depth": 5, "combination": "OTS", "gain": 15.0},
    ]
    done = {row["combination"]: row for row in ablation.add_overlap(rows)}
    # A pair overlaps: 2 x 3 = 6 predicted, 3 delivered, so half the saving was shared.
    assert done["OT-"]["product"] == 6.0
    assert done["OT-"]["delivered"] == 0.5
    # A pair that is exactly independent delivers 1.0 and must not be rounded away.
    assert done["O-S"]["product"] == 10.0
    assert done["O-S"]["delivered"] == 1.0
    # The triple multiplies all three singles, never the pairs.
    assert done["OTS"]["product"] == 30.0
    assert done["OTS"]["delivered"] == 0.5


def test_a_combination_of_fewer_than_two_switches_has_no_product():
    """A single switch's "product" would just be itself, which would put a meaningless
    1.0 in a column a reader is meant to read as overlap."""
    rows = [
        {"depth": 4, "combination": "---", "gain": 1.0},
        {"depth": 4, "combination": "O--", "gain": 2.0},
        {"depth": 4, "combination": "-T-", "gain": 3.0},
        {"depth": 4, "combination": "--S", "gain": 4.0},
        {"depth": 4, "combination": "OT-", "gain": 5.0},
    ]
    done = {row["combination"]: row for row in ablation.add_overlap(rows)}
    assert done["---"]["product"] == "" and done["---"]["delivered"] == ""
    assert done["O--"]["product"] == "" and done["O--"]["delivered"] == ""
    assert done["OT-"]["product"] == 6.0


def test_the_overlap_is_computed_within_one_depth_and_never_across_depths():
    """Gains at different depths are different numbers; mixing them would silently make
    the deepest row's product depend on the shallowest row's gain."""
    rows = [
        {"depth": 4, "combination": "O--", "gain": 2.0},
        {"depth": 4, "combination": "-T-", "gain": 2.0},
        {"depth": 4, "combination": "OT-", "gain": 4.0},
        {"depth": 7, "combination": "O--", "gain": 10.0},
        {"depth": 7, "combination": "-T-", "gain": 10.0},
        {"depth": 7, "combination": "OT-", "gain": 50.0},
    ]
    done = {(row["depth"], row["combination"]): row for row in ablation.add_overlap(rows)}
    assert done[(4, "OT-")]["product"] == 4.0
    assert done[(7, "OT-")]["product"] == 100.0


def test_the_committed_matrix_shows_overlap_where_P33_says_it_does():
    """Move ordering against anything else overlaps: the combined gain is below the
    product. This is the claim the ablation chapter closes on."""
    rows = {(int(r["depth"]), r["combination"]): r for r in _matrix_rows()}
    for depth in (5, 6, 7):
        delivered = float(rows[(depth, "O-S")]["delivered"])
        assert 0.4 < delivered < 1.0, (depth, delivered)


def test_the_table_and_negascout_are_the_one_pair_that_does_better_than_its_product():
    """Measured, and it contradicted the prediction written down before the run.

    NegaScout searches most nodes with a null window, which cannot store an exact value, so
    the table's entries stop being able to ANSWER a node and start supplying a MOVE instead
    -- a cutoff the table never used to cause. The counters say so directly: the
    transposition move's share of cutoffs goes from about a tenth of a percent to several
    percent as soon as scouting is on. Pinned here so a later change cannot quietly undo it.
    """
    rows = {(int(r["depth"]), r["combination"]): r for r in _matrix_rows()}
    for depth in (4, 5, 6):
        assert float(rows[(depth, "-TS")]["delivered"]) > 1.0, depth
        plain = float(rows[(depth, "-T-")]["tt_move_cutoff_rate"])
        scouted = float(rows[(depth, "-TS")]["tt_move_cutoff_rate"])
        assert scouted > 5 * plain, (depth, plain, scouted)


def test_every_combination_returned_the_frozen_reference_value():
    """Not re-run here -- the sweep asserts it as it goes. What is checked is that the
    committed file records a full grid, so no combination was quietly skipped."""
    rows = _matrix_rows()
    depths = sorted({int(row["depth"]) for row in rows})
    assert depths == [4, 5, 6, 7]
    for depth in depths:
        keys = {row["combination"] for row in rows if int(row["depth"]) == depth}
        assert len(keys) == 8, (depth, keys)


def test_the_parallel_control_is_a_speedup_curve_that_starts_at_one():
    rows = list(csv.DictReader((RESULTS / "parallel_control.csv").open(encoding="utf-8")))
    assert [int(row["workers"]) for row in rows] == [1, 2, 4, 6, 8]
    assert float(rows[0]["speedup"]) == 1.0
    # Identical tasks, so the only thing it can measure is the machine.
    assert len({int(row["tasks"]) for row in rows}) == 1


# --------------------------------------------------------------- the figures, as numbers


def _matrix_rows():
    """The committed matrix, with the numeric columns coerced back to numbers.

    `csv.DictReader` hands back strings, and `check_against_committed` keys on `(int, str)`
    -- so feeding it raw rows makes every lookup miss and the check passes by doing
    nothing. That is exactly what the first version of this file did, and the negative
    control two tests below is what caught it.
    """
    rows = []
    for row in csv.DictReader((RESULTS / "ablation_matrix.csv").open(encoding="utf-8")):
        row = dict(row)
        row["depth"] = int(row["depth"])
        row["positions"] = int(row["positions"])
        row["nodes"] = int(row["nodes"])
        rows.append(row)
    return rows


def test_every_figure_declares_a_chapter_and_real_source_files():
    """BUILD_PLAN section 9 maps chapters to artifacts; anything landing nowhere should not
    be produced at all."""
    for builder in plots.FIGURES_DATA + plots.TABLES_DATA:
        data = builder()
        assert data["chapter"], data["slug"]
        assert data["sources"], data["slug"]
        for source in data["sources"]:
            assert (RESULTS / source).exists(), (data["slug"], source)


def test_every_figure_names_its_population():
    """Thesis point P11: the empty board is alpha-beta's best case and minimax's worst, so
    a figure that does not say which population it is drawn from is not a result."""
    words = ("empty board", "reference position", "positions", "games", "round robin")
    for builder in plots.FIGURES_DATA:
        data = builder()
        caption = data["caption"].lower()
        assert any(word in caption for word in words), data["slug"]


def test_every_charted_series_also_appears_in_that_figures_table():
    """The rule that stops a figure being the only place a number lives."""
    for builder in plots.FIGURES_DATA:
        data = builder()
        assert data["rows"], data["slug"]
        assert data["columns"], data["slug"]
        for row in data["rows"]:
            assert set(row) >= set(data["columns"]) - {""}, data["slug"]


def test_node_counts_are_normalised_by_a_sample_size_that_changes_with_depth():
    """The reference set holds 200 positions to depth 5, then 40 and 10. Plotting totals
    would put a step in the curve that is an artefact of the sample and nothing else."""
    data = plots.figure1_data()
    rows = {row["depth"]: row for row in data["rows"]}
    committed = {
        int(r["depth"]): r
        for r in csv.DictReader((RESULTS / "alphabeta_by_position.csv").open(encoding="utf-8"))
    }
    for depth in (5, 6, 7):
        n = int(committed[depth]["positions"])
        expected = round(int(committed[depth]["minimax_nodes"]) / n, 1)
        assert rows[depth]["pos_minimax"] == expected, depth
    # The sample really does change, or the test above proves nothing.
    assert {rows[5]["positions"], rows[6]["positions"], rows[7]["positions"]} == {200, 40, 10}


def test_the_empty_board_panel_is_not_divided_by_anything():
    """One position means one search; dividing it by a sample size would be a lie about
    what was measured."""
    data = plots.figure1_data()
    rows = {row["depth"]: row for row in data["rows"]}
    committed = {
        int(r["depth"]): int(r["nodes"])
        for r in csv.DictReader((RESULTS / "minimax_nodes_by_depth.csv").open(encoding="utf-8"))
    }
    for depth, nodes in committed.items():
        assert rows[depth]["empty_minimax"] == nodes, depth


def test_the_branching_factor_chain_is_continuous_across_the_four_phases():
    """Each phase's baseline is the previous phase's optimised search, so the chain has no
    gaps and the "fraction of what was left" column means what it says."""
    data = plots.figure2_data()
    values = [row["d7"] for row in data["rows"]]
    assert values == sorted(values, reverse=True), values
    assert values[-1] > plots.SQRT7, "an effective branching factor below the floor"
    fractions = [row["fraction_of_remainder_closed"] for row in data["rows"][1:]]
    assert all(0.0 < fraction < 1.0 for fraction in fractions), fractions


def test_each_phase_closes_a_fraction_of_what_was_left_not_of_the_original_gap():
    """The three numbers thesis point P33 quotes, recomputed from the committed CSVs.

    Pinned, because "a fraction of the remainder" and "a fraction of the original gap" are
    both plausible-looking columns and only one of them supports the argument that the
    optimisations are racing at a floor. Measured against the original gap the same data
    reads 44% / 61% / 73%, which tells the opposite story.

    NOTE, and it is a correction rather than a disagreement: P33 states the third figure as
    **26%**, which comes from rounding the effective branching factor to 3.06 before
    subtracting the floor. The committed CSV holds 3.0642, and the exact arithmetic gives
    **27.2%**. The first two figures reproduce exactly (44.0%, 30.8%). No committed file was
    changed; the rounded claim is simply half a point optimistic about Phase 8 and a point
    pessimistic about Phase 10.
    """
    rows = plots.figure2_data()["rows"]
    closed = [row["fraction_of_remainder_closed"] for row in rows[1:]]
    assert closed[0] == pytest.approx(0.440, abs=0.005), closed
    assert closed[1] == pytest.approx(0.308, abs=0.005), closed
    assert closed[2] == pytest.approx(0.272, abs=0.005), closed


def test_the_branching_factor_stages_match_the_committed_sweeps():
    ordering = {
        (int(r["depth"]), r["ordering"]): float(r["ebf"])
        for r in csv.DictReader((RESULTS / "ordering_by_position.csv").open(encoding="utf-8"))
    }
    data = plots.figure2_data()
    assert data["rows"][0]["d7"] == ordering[(7, "center_first")]
    assert data["rows"][1]["d7"] == ordering[(7, "center_first+killers+history")]


def test_the_published_P33_slice_reproduces_the_committed_claim():
    """P33 states 82% / 80% / 68% at depths 5 / 6 / 7, measured in Phase 10 from a
    different file. Re-deriving it from the new matrix is a cross-check on both."""
    rows = {row["depth"]: row for row in plots.p33_slice()["rows"]}
    assert rows[5]["delivered"] == pytest.approx(0.82, abs=0.01)
    assert rows[6]["delivered"] == pytest.approx(0.80, abs=0.01)
    assert rows[7]["delivered"] == pytest.approx(0.68, abs=0.01)


def test_an_elo_band_is_carried_as_two_numbers_because_it_is_not_symmetric():
    data = plots.figure3_data()
    for row in data["rows"]:
        assert row["elo_low"] <= row["elo"] <= row["elo_high"], row["label"]
        assert row["band_minus"] > 0 and row["band_plus"] > 0, row["label"]
        # The two halves are the distances to the two percentiles, not half the width.
        assert row["band_minus"] == pytest.approx(row["elo"] - row["elo_low"], abs=0.05)
        assert row["band_plus"] == pytest.approx(row["elo_high"] - row["elo"], abs=0.05)
    assert [row["label"] for row in data["rows"]][0] == "random"
    # A percentile interval really is lopsided here, so collapsing it to a single +- would
    # be a claim the bootstrap never made.
    assert any(
        abs(row["band_minus"] - row["band_plus"]) > 1.0 for row in data["rows"]
    ), "no asymmetric band -- this test would not notice a symmetric one"


def test_the_elo_caption_states_the_blind_spot_of_its_own_interval():
    """Thesis point P36: a clean sweep is resampled from p = 0 and gets a band far narrower
    than its real uncertainty. What makes that rating finite is the prior, not the band."""
    caption = plots.figure3_data()["caption"]
    assert "prior" in caption and "0.0%" in caption


def test_the_parallel_figure_carries_its_control_and_says_who_wins():
    data = plots.figure4_data()
    verdicts = {row["workers"]: row["harness_beats_control"] for row in data["rows"]}
    assert verdicts[8] == "yes", "the claim that the ceiling is the machine rests on this"
    assert all(row["ideal_speedup"] == float(row["workers"]) for row in data["rows"])


def test_the_verdict_is_computed_from_the_numbers_and_not_asserted():
    """P35 claims the harness beats the control at every worker count. Re-measured with
    its own committed CSV it does so at 4, 6 and 8 -- and NOT at 2, where the two are
    1.2% apart and the control is nominally ahead.

    That "no" is pinned deliberately. A column that said "yes" everywhere would be
    publishing a conclusion the data does not support, and the mutation check found that
    the previous version of this test could not tell the difference.
    """
    data = plots.figure4_data()
    verdicts = {row["workers"]: row["harness_beats_control"] for row in data["rows"]}
    assert verdicts[2] == "no", verdicts
    assert verdicts[4] == verdicts[6] == verdicts[8] == "yes", verdicts
    rows = {row["workers"]: row for row in data["rows"]}
    for workers, row in rows.items():
        if row["harness_speedup"] == "" or row["control_speedup"] == "":
            continue
        expected = "yes" if row["harness_speedup"] > row["control_speedup"] else "no"
        assert row["harness_beats_control"] == expected, workers


def test_every_rate_that_gets_a_bar_carries_a_sigma():
    """A rate without its standard error is not a result."""
    for row in plots.figure5_data()["rows"]:
        assert row["preserved_sigma"] >= 0.0
        assert row["preserved_sigma"] == pytest.approx(
            plots._binomial_sigma(row["preserved_rate"], row["positions"]), abs=5e-5
        )
    for row in plots.figure6_data()["rows"]:
        assert row["optimal_sigma"] > 0.0 or row["optimal_rate"] in (0.0, 1.0)


def test_a_binomial_sigma_is_zero_at_certainty_and_largest_at_a_half():
    assert plots._binomial_sigma(0.0, 100) == 0.0
    assert plots._binomial_sigma(1.0, 100) == 0.0
    assert plots._binomial_sigma(0.5, 100) == pytest.approx(0.05)
    # It shrinks as the square root of the sample, which is the reason Phase 12 exists.
    assert plots._binomial_sigma(0.5, 400) == pytest.approx(0.025)


def test_the_benchmark_figure_keeps_the_outcome_split_it_exists_for():
    """Thesis point P37: a lost position cannot contain a blunder, and MCTS collapses on
    lost positions. Averaging the three outcomes would delete both findings."""
    data = plots.figure6_data()
    sets = dict(data["sets"])
    assert set(sets) == {"end-easy", "middle-easy"}
    for grid in sets.values():
        for agent in plots.AGENT_ORDER:
            for outcome in plots.OUTCOMES:
                assert (agent, outcome) in grid, (agent, outcome)
    lost_mcts = sets["middle-easy"][("mcts", "loss")]["optimal_rate"]
    lost_greedy = sets["middle-easy"][("greedy", "loss")]["optimal_rate"]
    assert lost_mcts < lost_greedy, (lost_mcts, lost_greedy)


def test_a_markdown_table_has_one_row_per_record_and_a_rule():
    text = plots.markdown_table(["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    lines = text.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"
    assert len(lines) == 4


def test_the_results_document_holds_every_figure_and_every_table():
    text = plots.results_markdown()
    for builder in plots.FIGURES_DATA + plots.TABLES_DATA:
        assert builder()["slug"] in text
    for name in plots.OMITTED:
        assert name in text, name


def test_both_languages_define_the_same_labels():
    """A missing key would not fail until a figure was drawn in that language."""
    assert set(plots.LATIN) == set(plots.CYRILLIC)
    assert plots.LATIN != plots.CYRILLIC


def test_the_cyrillic_labels_are_actually_cyrillic():
    """Guards against a half-translated dictionary, which would look fine in a diff."""
    for key, value in plots.CYRILLIC.items():
        letters = [character for character in value if character.isalpha()]
        if not letters:
            continue
        cyrillic = [character for character in letters if "\u0400" <= character <= "\u04ff"]
        assert len(cyrillic) > len(letters) / 2, (key, value)


def test_importing_the_plot_module_does_not_import_matplotlib():
    """The drawing library must stay out of anything the app or a worker touches.

    Checked in a fresh interpreter for the same reason the pygame assertion is: inside this
    suite matplotlib may already be loaded by the rendering tests below.
    """
    program = (
        "import sys; import experiments.plots;"
        "assert 'matplotlib' not in sys.modules,"
        " sorted(m for m in sys.modules if 'matplotlib' in m);"
        "print('clean')"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "clean" in done.stdout


# ------------------------------------------------------- the drawing, where it is testable
# Everything below needs matplotlib. It is NOT in `requirements.txt` -- the application runs
# without it -- so these skip rather than fail on a bare machine.


def test_both_scripts_render_every_glyph_we_need():
    """Proven by looking at pixels, not by assuming a font has what it says it has.

    A font without a glyph does not raise -- it draws `.notdef`, an empty rectangle, and it
    draws the same rectangle every time. So each character is compared against a character
    the font is known to be missing.
    """
    pytest.importorskip("matplotlib")
    for glyphs in (plots.LATIN_GLYPHS, plots.CYRILLIC_GLYPHS):
        verdict = plots.prove_glyphs(glyphs)
        missing = [character for character, ok in verdict.items() if not ok]
        assert not missing, missing


def test_the_glyph_proof_would_notice_a_character_the_font_lacks():
    """The control for the control. If this passed, the test above would pass for a font
    with no Cyrillic in it whatsoever."""
    pytest.importorskip("matplotlib")
    verdict = plots.prove_glyphs(plots.MISSING_GLYPH)
    assert verdict[plots.MISSING_GLYPH] is False


def test_the_figures_carry_no_timestamp_and_redraw_to_the_same_bytes():
    """Two halves, and neither is sufficient on its own.

    Comparing two back-to-back runs is a FALSE PASS: matplotlib stamps a `CreationDate`
    into a PDF that only changes once a SECOND, so two runs a millisecond apart agree
    perfectly while tomorrow's run would not. The reproducibility claim therefore also
    requires that the field be absent, which is a property of a single file.
    """
    pytest.importorskip("matplotlib")
    assert plots.check_determinism("latin") == []


def test_the_timestamp_detector_actually_detects_a_timestamp(tmp_path):
    """The other half of the gate above, and the mutation check is why it exists: a
    `stamps_in` that always returned nothing left every figure certified and no test
    noticed."""
    pytest.importorskip("matplotlib")
    plt = plots._pyplot()
    figure = plt.figure(figsize=(1, 1))
    stamped = tmp_path / "stamped.pdf"
    clean = tmp_path / "clean.pdf"
    figure.savefig(stamped, format="pdf")                       # default: dated
    figure.savefig(clean, format="pdf", metadata=plots.PDF_METADATA)
    plt.close(figure)
    assert plots.stamps_in(stamped), "a dated PDF was reported as clean"
    assert plots.stamps_in(clean) == []


def test_a_written_figure_has_both_a_vector_and_a_raster_form():
    """PDF for the document, PNG for the slides, from one call."""
    pytest.importorskip("matplotlib")
    figures = ROOT / "data" / "figures"
    for builder in plots.FIGURES_DATA:
        slug = builder()["slug"]
        assert (figures / f"{slug}.pdf").exists(), slug
        assert (figures / f"{slug}.png").exists(), slug
        assert (figures / f"{slug}_cyr.pdf").exists(), slug
        assert (figures / f"{slug}_cyr.png").exists(), slug
