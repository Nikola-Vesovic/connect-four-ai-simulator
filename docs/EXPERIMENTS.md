# Experiment Guide

Run commands from the repository root with its Python environment activated.
The [README](../README.md) covers installation and the interactive simulator.

## Reproducibility

Keep the agent configuration, board representation, seed, opening policy,
evaluation, move ordering, transposition strategy, and budget with each result.
Use fixed depth, node, or simulation budgets for repeatable work; elapsed time
will still vary. A simulation can traverse many positions and is not one node.

Results are split by which player moved first. Randomized openings diversify
matches between deterministic agents. Stochastic agents receive derived seeds,
and agent state is reset between games. Parallel checks compare result digests,
not timing columns.

Committed CSVs are the thesis record. Several commands below **write into
data/results/** or regenerate figures. Run new research in a separate checkout,
or review the resulting Git diff before keeping changes. In particular, the
reference generator replaces the frozen baseline used by regression tests.

## Matches

```sh
python -m experiments.match --a greedy --b random --games 100 --seed 2026
python -m experiments.match --a alphabeta --b greedy --games 20 --opening-plies 4
python -m experiments.match --help
```

The CLI exposes named agent configurations. The GUI's per-seat checkboxes and
the command-line configuration are two frontends to the same search code.

## Tournaments

[JSON configurations](../data/configs/) specify agents, budgets, seeds, opening
length, games per pairing, and bootstrap settings.

```sh
python -m experiments.runner --config data/configs/smoke.json --workers 2 --no-write
python -m experiments.runner --config data/configs/smoke.json --check 2 --no-write
```

The small smoke configuration is for validation, not a playing-strength claim.
The full ladder contains six agents and 500 games for each of 15 pairings:

```sh
python -m experiments.runner --config data/configs/ladder.json --workers 4 --no-write
```

Remove `--no-write` to save results. Full experiments may take considerable
time. `--workers 1` is truly serial; larger counts use spawned worker processes.
Time-budgeted tournaments require `--workers 1` to avoid load-biased comparisons.

Elo estimates are fitted jointly with a Bradley-Terry model; draws count as half
a point. Bootstrap intervals describe the sampled tournament and have limited
information about outcomes never observed. They are not universal Elo ratings.

Use the same Python version when reproducing exact bootstrap bounds. Python 3.11
samples individual Bernoulli trials; Python 3.12+ uses the standard library's
binomial sampler. Both sample the same distribution, but their seeded results
need not be identical across versions. Stored thesis results are unchanged.

## Minimax Reference Checks

The committed reference file stores values from the unoptimized baseline.
Optimized fixed-depth searches must preserve those values, even when their
visited nodes or selected tied moves differ.

```sh
python -m experiments.reference --verify --agent alphabeta --max-depth 4
python -m experiments.reference --verify --agent alphabeta --ordering center_first
python -m experiments.reference --verify --agent alphabeta --table no_table
python -m experiments.reference --verify --agent alphabeta --scout no_scout
```

`--agent negamax` also checks unpruned node counts. MCTS has different semantics
and is instead tested through invariants, seeded behavior, tactics, and
statistical performance. A time-interrupted search may complete a different
depth, so fixed-depth agreement is not a claim about equal-time move choices.

## Exact-Position Benchmarks

Download the optional inputs before running these commands:

```sh
python -m experiments.download_testsets
python -m experiments.benchmark --verify --sets end-easy --positions 24 --workers 2
python -m experiments.benchmark --score --sets end-easy --positions 24 --workers 2
python -m experiments.benchmark --check 2 --sets end-easy --positions 24
```

The downloader checks fixed SHA-256 hashes and rejects changed input. Files are
ignored by Git. See the [source and rights note](../data/testsets/gamesolver/SOURCE.txt).
HTTP transport is untrusted; checksums establish consistency with the recorded
inputs, not proof of their original authenticity or redistribution permission.
The loader additionally validates position syntax, legal play, and score bounds.

The oracle first verifies its scores against the published positions, then
evaluates legal children to identify value-preserving moves. Unsettled positions
are counted and excluded, not silently assigned an answer. Measures include:

- **Optimal-move rate:** membership in the set preserving the exact score.
- **Blunder rate:** a move changes the theoretical win/draw/loss outcome.
- **Value loss:** score loss, with mate-distance loss separated from outcome loss.

Rates are split by outcome and whether the position distinguishes the agents.
For example, an already lost position cannot incur an outcome-changing blunder.
`--write` saves benchmark CSVs; omitting it leaves committed measurements intact.

## Scaling and Ablations

Use `python -m experiments.scaling --help` for the full list. Representative
sweeps cover each optimization on empty boards, reference positions, and games:

```sh
python -m experiments.scaling --pruning
python -m experiments.scaling --positions
python -m experiments.scaling --ordering
python -m experiments.scaling --table
python -m experiments.scaling --deepening
python -m experiments.scaling --scout
python -m experiments.scaling --mcts-cost
python -m experiments.scaling --mcts-rollout-time
python -m experiments.scaling --mcts-vs-alphabeta
python -m experiments.ablation --matrix
```

The ablation matrix evaluates all eight combinations of move ordering, a
transposition table, and NegaScout. Savings can overlap, so their individual
speedups should not simply be multiplied. These sweeps write result files.

## Figures and Screenshots

Plotting reads existing CSVs; it does not rerun searches. The thesis figures
retain their Serbian Latin and Cyrillic labels.

```sh
python -m pip install -r requirements-figures.txt
python -m experiments.plots --all --both-languages
python -m experiments.plots --check
python -m experiments.plots --glyphs
python -m experiments.screenshots
```

Figures are saved as PNG and PDF alongside a generated results table.
`--check` compares repeated renders within the current environment; it is not a
promise of identical image bytes across operating systems or font versions.
Screenshots replay actual games and include move/statistics provenance; measured
wall-clock times are expected to differ between captures.

## Board Conventions

Internally, row 0 is the bottom and column 0 is the left. A bitboard uses
`bit = column * 7 + row`, with a sentinel bit above each six-cell column.
Moves are zero-based column indices. Benchmark move strings such as `"4455"`
use one-based columns, as does the screenshot provenance file.