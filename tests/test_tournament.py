"""Phase 12: the tournament harness.

There is no course algorithm here, so there is no value to check against a frozen
reference. What replaces it is three different kinds of assertion, and they should not be
blurred:

* **identities** -- a shard is the serial run restricted, and merging shards is
  associative, so the answer cannot depend on how the work was cut up or on which worker
  finished first;
* **a refusal** -- a wall-clock budget in parallel raises rather than warns, because a
  timed match under load measures the load (thesis point P24);
* **derivations** -- the Elo fit reproduces a score it was given, the confidence band
  reproduces from its seed, and both are checked against numbers computed by hand rather
  than against themselves.

The expensive one is `test_a_tournament_reproduces_between_one_worker_and_eight`, which
starts real spawned interpreters. It is the gate of the phase and is worth its seconds.
"""

from __future__ import annotations

import math
import pickle
import random

import pytest

from core.bitboard import Bitboard
from experiments.match import Tally, play_match, standard_error
from experiments.runner import (
    CONFIGS,
    TimedMatchRefused,
    TournamentConfig,
    check_parallelisable,
    elo_rows,
    pairing_rows,
    run,
)
from experiments.tournament import (
    AgentSpec,
    Pairing,
    PairingOutcome,
    Shard,
    _percentile,
    bootstrap_elo,
    elo_difference,
    elo_difference_error,
    expected_score,
    first_mover_tally,
    fit_elo,
    merge,
    play_shard,
    resolvable_margin,
    round_robin,
    seat_records,
    shard_bounds,
    shards_for,
)
from search.stats import Budget

RANDOM = AgentSpec("random", "random")
GREEDY = AgentSpec("greedy", "greedy")
MM2 = AgentSpec("mm2", "minimax", {"depth": 2})


def tiny(games: int = 4, budget: Budget = Budget()) -> TournamentConfig:
    return TournamentConfig(
        name="tiny",
        agents=(RANDOM, GREEDY, MM2),
        games=games,
        seed=2026,
        opening_plies=4,
        budget=budget,
        elo_resamples=50,
    )


# -------------------------------------------------------------- the shard is an identity


def test_a_shard_is_the_match_restricted_to_its_own_games():
    """The claim the whole phase rests on: cutting a match up changes nothing."""
    whole = play_match(GREEDY.build(), RANDOM.build(), games=24, seed=7, opening_plies=4)
    parts = [
        play_match(
            GREEDY.build(), RANDOM.build(), games=24, seed=7, opening_plies=4,
            indices=range(first, last),
        )
        for first, last in shard_bounds(24, 5)
    ]
    assert merge(parts).fingerprint() == whole.fingerprint()
    assert merge(parts).digest() == whole.digest()


def test_merging_is_independent_of_the_order_the_shards_finish_in():
    parts = [
        play_match(
            GREEDY.build(), RANDOM.build(), games=16, seed=7, opening_plies=4,
            indices=range(first, last),
        )
        for first, last in shard_bounds(16, 4)
    ]
    shuffled = list(parts)
    random.Random(1).shuffle(shuffled)
    assert merge(shuffled).digest() == merge(parts).digest()


def test_merge_refuses_shards_of_different_matches():
    one = play_match(GREEDY.build(), RANDOM.build(), games=4, seed=7)
    other = play_match(GREEDY.build(), RANDOM.build(), games=4, seed=8)
    with pytest.raises(ValueError):
        merge([one, other])


def test_shard_bounds_cover_every_game_exactly_once():
    for games in range(0, 33):
        for shards in range(1, 10):
            bounds = shard_bounds(games, shards)
            covered = [index for first, last in bounds for index in range(first, last)]
            assert covered == list(range(games)), (games, shards)


def test_shard_bounds_never_split_a_paired_opening():
    """Games 2k and 2k+1 are the two seatings of one opening. A boundary between them
    would leave no shard that is a valid sub-match on its own."""
    for games in range(0, 33):
        for shards in range(1, 10):
            for first, last in shard_bounds(games, shards):
                assert first % 2 == 0
                assert last % 2 == 0 or last == games


def test_asking_for_more_shards_than_there_are_pairs_degrades_quietly():
    assert len(shard_bounds(6, 50)) == 3
    assert shard_bounds(1, 50) == [(0, 1)]
    assert shard_bounds(0, 4) == []


def test_shards_are_numbered_in_the_order_they_must_be_merged():
    config = tiny(games=8)
    shards = shards_for(config.pairings(), 3)
    assert [shard.order for shard in shards] == list(range(len(shards)))


# ------------------------------------------------------- what crosses the process boundary


def test_an_agent_spec_is_data_and_never_an_agent():
    """Trap (c): sharing one agent object across processes is wrong and pickling one is
    worse. What travels is the description; the child builds its own."""
    spec = AgentSpec("ab", "negamax", {"depth": 3, "deepening": "iterative+pv+order"})
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    first, second = restored.build(), restored.build()
    assert first is not second
    assert first.name == second.name == "alphabeta-d3-iterative+pv+order"


def test_a_shard_survives_the_round_trip_and_stays_small():
    shard = Shard(tiny().pairings()[0], 0, 0, 4)
    payload = pickle.dumps(shard)
    assert pickle.loads(payload) == shard
    assert len(payload) < 2000, len(payload)


def test_an_empty_shard_is_a_legal_shard():
    """What `boundary_cost` measures with: a real shard that plays no games."""
    empty = play_shard(Shard(tiny().pairings()[0], 0, 0, 0))
    assert empty.games == 0
    assert pickle.loads(pickle.dumps(empty)).games == 0


def test_an_unknown_algorithm_is_refused_when_the_spec_is_built():
    with pytest.raises(ValueError, match="unknown algorithm"):
        AgentSpec("x", "alphabeta")


# ----------------------------------------------------------------------- the round robin


def test_round_robin_pairs_every_agent_once_and_never_with_itself():
    pairs = round_robin((RANDOM, GREEDY, MM2))
    assert len(pairs) == 3
    assert all(a.label != b.label for a, b in pairs)
    assert {(a.label, b.label) for a, b in pairs} == {
        ("random", "greedy"), ("random", "mm2"), ("greedy", "mm2")
    }


def test_round_robin_refuses_duplicate_labels():
    with pytest.raises(ValueError, match="unique"):
        round_robin((GREEDY, AgentSpec("greedy", "random")))


def test_every_pairing_keeps_the_paired_opening():
    """Deterministic agents replay one game, so `opening_plies` is what makes a match a
    sample at all (Phase 5). A round robin must not lose it on the way through."""
    config = tiny(games=6)
    for pairing in config.pairings():
        assert pairing.opening_plies == config.opening_plies
        assert pairing.seed == config.seed
        assert pairing.games == config.games


def test_the_deterministic_repeat_detector_still_fires_on_a_merged_result():
    """`repeats_one_game` is the cheapest bug detector in the project and merging must
    not blunt it."""
    config = TournamentConfig("p", (GREEDY, MM2), games=8, opening_plies=0,
                              elo_resamples=10)
    finished = run(config, 1, progress=False)
    assert finished.results[("greedy", "mm2")].repeats_one_game


# ------------------------------------------------------------------------- the refusal


def test_a_timed_budget_in_parallel_is_refused_not_warned():
    """Trap (a). Phase 9 measured the cost of ignoring this: the same 540 games said
    44.3% under a clock and 48.5% under a node budget, and only the second had a
    control."""
    with pytest.raises(TimedMatchRefused):
        check_parallelisable(tiny(budget=Budget(time=0.05)), 2)
    with pytest.raises(TimedMatchRefused):
        check_parallelisable(tiny(budget=Budget(nodes=4000, time=0.05)), 8)
    with pytest.raises(TimedMatchRefused):
        run(tiny(budget=Budget(time=0.01)), 4, progress=False)


def test_a_timed_budget_is_allowed_in_one_process_and_says_it_does_not_reproduce():
    config = tiny(games=2, budget=Budget(time=0.001))
    check_parallelisable(config, 1)
    assert config.budget_unit == "time"
    finished = run(config, 1, progress=False)
    assert "does not reproduce" in _report(finished)


def test_a_node_budget_is_parallelisable_and_says_which_unit_it_used():
    config = tiny(budget=Budget(nodes=500))
    check_parallelisable(config, 8)
    assert config.budget_unit == "nodes"
    assert {row["budget_unit"] for row in pairing_rows(run(config, 1, progress=False))} == {
        "nodes"
    }


def _report(finished):
    from experiments.runner import format_report

    return format_report(finished, elo_rows(finished))


# ---------------------------------------------------------------------------- the gate


def test_a_tournament_reproduces_between_one_worker_and_eight():
    """The gate of the phase, in its sharp form.

    One worker means no multiprocessing at all -- literally what every phase before this
    one did -- so this compares the previous state of the project against eight spawned
    interpreters and demands the same bytes. It holds because a game is a pure function
    of its index: `game_seeds` fixes the seed pair from the master seed, and
    `Agent.new_game(seed)` throws away the only two pieces of state that outlive a game,
    MCTS's tree and a persistent transposition table.
    """
    config = tiny(games=6)
    serial = run(config, 1, progress=False)
    parallel = run(config, 8, progress=False)
    again = run(config, 1, progress=False)
    assert serial.digest() == parallel.digest() == again.digest()
    assert parallel.shards > serial.shards
    for key, match in serial.results.items():
        assert match.fingerprint() == parallel.results[key].fingerprint()


# ------------------------------------------------------------------------------- the Elo


def test_the_fit_reproduces_the_score_it_was_given():
    """The defining property of the maximum-likelihood fit for one pairing: the rating
    gap it returns must predict exactly the score that was observed."""
    outcome = PairingOutcome("a", "b", 750, 0, 250)
    ratings = fit_elo([outcome], ["a", "b"], prior=1e-9)
    gap = ratings["a"] - ratings["b"]
    assert gap == pytest.approx(400 * math.log10(3), abs=1e-3)
    assert expected_score(gap) == pytest.approx(0.75, abs=1e-6)


def test_draws_are_worth_half_a_point_to_each_side():
    """A convention, not a discovery, and the table has to say so: the fit models the
    score, so a pairing that is 60 wins and 40 draws and one that is 80 wins and 20
    losses are the same 80% to it."""
    a = fit_elo([PairingOutcome("a", "b", 60, 40, 0)], ["a", "b"], prior=1e-9)
    b = fit_elo([PairingOutcome("a", "b", 80, 0, 20)], ["a", "b"], prior=1e-9)
    assert a["a"] - a["b"] == pytest.approx(b["a"] - b["b"], abs=1e-6)


def test_elo_is_only_defined_up_to_a_constant_so_the_table_is_centred():
    ratings = fit_elo(
        [PairingOutcome("a", "b", 150, 10, 40), PairingOutcome("b", "c", 140, 10, 50),
         PairingOutcome("a", "c", 180, 5, 15)],
        ["a", "b", "c"],
    )
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-6)
    assert ratings["a"] > ratings["b"] > ratings["c"]


def test_a_clean_sweep_has_no_finite_rating_without_a_prior():
    """Written down before the tournament was run, not discovered in it: an agent that
    scores nothing against everyone has no maximum-likelihood rating at all."""
    outcomes = [PairingOutcome("a", "b", 200, 0, 0)]
    with pytest.raises(ValueError, match="prior"):
        fit_elo(outcomes, ["a", "b"], prior=0.0)
    ratings = fit_elo(outcomes, ["a", "b"], prior=2.0)
    assert all(math.isfinite(value) for value in ratings.values())
    assert ratings["a"] > ratings["b"]


def test_identical_agents_get_identical_ratings_and_a_band_that_covers_zero():
    outcomes = [
        PairingOutcome("a", "b", 50, 100, 50),
        PairingOutcome("b", "c", 50, 100, 50),
        PairingOutcome("a", "c", 50, 100, 50),
    ]
    ratings = fit_elo(outcomes, ["a", "b", "c"])
    assert all(value == pytest.approx(0.0, abs=1e-6) for value in ratings.values())
    band = bootstrap_elo(outcomes, ["a", "b", "c"], resamples=200, seed=2026)
    assert all(low < 0.0 < high for low, high in band.values())


def test_the_fit_and_its_band_both_reproduce_exactly():
    outcomes = [PairingOutcome("a", "b", 120, 30, 50), PairingOutcome("a", "c", 90, 10, 100),
                PairingOutcome("b", "c", 70, 20, 110)]
    labels = ["a", "b", "c"]
    assert fit_elo(outcomes, labels) == fit_elo(outcomes, labels)
    assert bootstrap_elo(outcomes, labels, resamples=100, seed=5) == bootstrap_elo(
        outcomes, labels, resamples=100, seed=5
    )
    other = bootstrap_elo(outcomes, labels, resamples=100, seed=6)
    assert other != bootstrap_elo(outcomes, labels, resamples=100, seed=5)


def test_a_wider_band_needs_fewer_games_and_a_narrower_one_more():
    """A confidence interval that did not shrink with the sample would be decoration."""
    labels = ["a", "b"]
    small = bootstrap_elo([PairingOutcome("a", "b", 30, 10, 20)], labels,
                          resamples=300, seed=3)
    large = bootstrap_elo([PairingOutcome("a", "b", 300, 100, 200)], labels,
                          resamples=300, seed=3)
    width = lambda band: band["a"][1] - band["a"][0]  # noqa: E731
    assert width(large) < width(small) / 2


def test_the_pairwise_error_uses_the_real_variance_of_a_draw():
    """It reuses `standard_error`, which counts a draw as a half rather than pretending
    the outcome is a coin flip; a drawish pairing is measured more precisely, not less."""
    drawish = Tally(20, 60, 20)
    decisive = Tally(50, 0, 50)
    assert drawish.score == decisive.score == 0.5
    assert standard_error(drawish) < standard_error(decisive)
    assert elo_difference_error(drawish) < elo_difference_error(decisive)
    assert elo_difference(0.5) == 0.0
    assert elo_difference_error(Tally(100, 0, 0)) == math.inf


def test_the_elo_gap_is_a_logistic_curve_and_not_a_win_rate():
    """Added after the mutation check: every assertion about `elo_difference` was at a
    score of 0.5, where a straight line through the origin agrees with the curve, so a
    mutant that replaced the whole thing with `400 * (score - 0.5)` survived."""
    assert elo_difference(0.75) == pytest.approx(400 * math.log10(3), abs=1e-9)
    assert elo_difference(0.25) == pytest.approx(-400 * math.log10(3), abs=1e-9)
    assert elo_difference(0.9) == pytest.approx(381.7, abs=0.1)
    assert elo_difference(1.0) == math.inf and elo_difference(0.0) == -math.inf
    assert expected_score(elo_difference(0.64)) == pytest.approx(0.64, abs=1e-9)
    # A win rate would make these equal; the curve makes the second far larger.
    assert elo_difference(0.95) - elo_difference(0.85) > 2 * (
        elo_difference(0.55) - elo_difference(0.45)
    )


def test_a_percentile_interpolates_instead_of_taking_an_extreme():
    """Also added after the mutation check: returning `min` and `max` instead of the
    2.5th and 97.5th percentile still produces a band that contains the estimate, still
    reproduces, and still narrows with the sample, so three band tests passed anyway.
    The sharp version asks for two confidence levels: a real percentile makes the 95%
    band strictly narrower than the full range of the resamples, an extreme does not."""
    values = [float(index) for index in range(101)]
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 1.0) == 100.0
    assert _percentile(values, 0.5) == 50.0
    assert _percentile(values, 0.025) == pytest.approx(2.5)
    assert _percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)

    outcomes = [PairingOutcome("a", "b", 120, 30, 50)]
    inner = bootstrap_elo(outcomes, ["a", "b"], resamples=400, seed=9, confidence=0.95)
    full = bootstrap_elo(outcomes, ["a", "b"], resamples=400, seed=9, confidence=1.0)
    assert full["a"][0] < inner["a"][0] and inner["a"][1] < full["a"][1]


def test_the_sizing_arithmetic_reproduces_the_numbers_this_project_has_quoted():
    """P24 sized 540 games at 4.3 points; P30 said about 1,100 for three."""
    assert resolvable_margin(540) == pytest.approx(4.3, abs=0.05)
    assert resolvable_margin(1100) == pytest.approx(3.0, abs=0.05)
    assert resolvable_margin(200) == pytest.approx(7.1, abs=0.05)


# ----------------------------------------------------------------------- the seat split


def test_every_pairing_is_filed_under_both_agents_and_split_by_seat():
    """Thesis point P1: an aggregate number per agent is not a measurement of strength."""
    finished = run(tiny(games=8), 1, progress=False)
    records = seat_records(finished.results, list(tiny().labels))
    for label, record in records.items():
        assert record.first.games == record.second.games, label
        assert record.overall.games == 2 * 8  # two opponents, eight games each


def test_the_second_seat_is_the_mirror_of_the_first_and_not_a_copy():
    """The Phase 4 trap, as a test: both tallies of a `MatchResult` are scored from A's
    side, so B's view of them is the mirror. Reading it the other way turns a 55.6%
    first-mover edge into 47.4%.

    Two agents, so there is exactly one pairing to compare against. The first draft used
    the three-agent config and asserted against a single match while `greedy` was also
    collecting games from a second pairing -- my test, not the code.
    """
    config = TournamentConfig("pair", (RANDOM, GREEDY), games=8, opening_plies=4,
                              elo_resamples=10)
    finished = run(config, 1, progress=False)
    match = finished.results[("random", "greedy")]
    records = seat_records(finished.results, ["random", "greedy"])
    assert records["random"].first == match.a_first
    assert records["random"].second == match.b_first
    assert records["greedy"].first == match.b_first.mirrored()
    assert records["greedy"].second == match.a_first.mirrored()
    assert records["greedy"].first.wins == match.b_first.losses


def test_the_first_mover_tally_counts_every_game_exactly_once():
    finished = run(tiny(games=8), 1, progress=False)
    first = first_mover_tally(finished.results.values())
    assert first.games == sum(m.games for m in finished.results.values())
    assert first.wins == sum(m.first_mover_wins for m in finished.results.values())


# ----------------------------------------------------------------------- the config file


def test_every_committed_config_loads_and_every_agent_in_it_can_play():
    """A config file is a promise that a tournament can be re-run from it.

    The probe carries a budget on **both** axes, and the first draft of this test did
    not: a config may name a 42-ply iterative agent or a million-simulation MCTS agent,
    and asking either of them for a move with no ceiling is asking it to solve Connect
    Four. `nodes` binds the iterative and sampling agents, `depth` binds the fixed-depth
    ones, and neither can be dropped.
    """
    probe = Budget(nodes=200, depth=2)
    for path in sorted(CONFIGS.glob("*.json")):
        config = TournamentConfig.load(path)
        assert config.games > 0 and config.agents
        assert len(set(config.labels)) == len(config.agents), path
        for spec in config.agents:
            move, stats = spec.build().choose_move(Bitboard(), probe)
            assert move in range(7), (path, spec.label)
            assert stats.move == move
        config.pairings()


def test_a_config_names_its_specs_the_way_the_committed_csvs_do():
    """One vocabulary for configs and results: a spec cannot be renamed in one place
    without the other noticing."""
    spec = AgentSpec("x", "negamax", {"depth": 5, "ordering": "center_first",
                                      "transposition": "no_table", "scouting": "no_scout"})
    assert spec.agent_name == "alphabeta-d5-center_first-no_table-no_scout"


def test_an_unknown_budget_axis_is_refused_rather_than_ignored(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"name":"b","games":2,"budget":{"seconds":1},"agents":'
                   '[{"label":"r","algorithm":"random"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown budget"):
        TournamentConfig.load(bad)


def test_the_run_digest_covers_every_pairing():
    """One digest for the whole tournament, so the gate is one comparison."""
    finished = run(tiny(games=6), 1, progress=False)
    changed = run(TournamentConfig("tiny", (RANDOM, GREEDY, MM2), games=6, seed=2027,
                                   opening_plies=4, elo_resamples=10), 1, progress=False)
    assert finished.digest() != changed.digest()
    assert len(finished.digest()) == 12


def test_the_pairing_table_reports_what_produced_every_number():
    """Thesis point P10, now with the budget and its unit beside it (P29)."""
    rows = pairing_rows(run(tiny(budget=Budget(nodes=500)), 1, progress=False))
    for row in rows:
        assert row["budget"] == "500 nodes" and row["budget_unit"] == "nodes"
        assert row["a_agent"] and row["b_agent"]
        assert row["wins"] + row["draws"] + row["losses"] == row["games"]


def test_the_elo_table_keeps_the_seat_split_beside_every_rating():
    rows = elo_rows(run(tiny(games=8), 1, progress=False))
    assert [row["label"] for row in rows] == sorted(
        (row["label"] for row in rows), key=lambda name: -next(
            r["elo"] for r in rows if r["label"] == name)
    )
    for row in rows:
        assert row["elo_low"] <= row["elo"] <= row["elo_high"]
        assert "score_first" in row and "score_second" in row
