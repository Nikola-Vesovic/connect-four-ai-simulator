# Connect Four — agenti pretrage (diplomski rad)

Minimax / alfa-beta / negascout vs. MCTS na igri Connect Four.
Plan rada: [../plan/BUILD_PLAN.md](../plan/BUILD_PLAN.md)

## Struktura

```
connect4/
  core/          rules.py (ugovor), types.py (+ Score skala), board_naive.py, bitboard.py [1-2, 5]
  agents/        base.py (interfejs), random_agent.py, greedy_agent.py                    [4]
                 minimax.py (MAX/MIN, ogranicen dubinom)                                  [5]
                 alphabeta, mcts, oracle                                                  [6-11]
  search/        stats.py (Budget + SearchStats), ordering.py (centar-prvo)               [4]
                 evaluation.py (pretnje, centar, parnost -- svaka iskljuciva)             [5]
                 transpoziciona tabela, tajmer                                            [8-9]
  gui/           pygame prikaz                                                            [3, posle 9]
  experiments/   match.py (mec izmedju dva agenta, sa semenom)                            [4]
                 reference.py (zamrznute vrednosti), scaling.py (cvorovi i kriva)         [5]
                 turniri, benchmark, ablacije, grafici                                    [12-14]
  data/          testsets/ (referentne vrednosti), results/ (CSV), figures/
  tests/         pytest
```

## Merenja

```sh
python -m experiments.scaling --nodes      # cvorovi po dubini, prazna tabla
python -m experiments.scaling --curve      # rezultat protiv greedyja po dubini
python -m experiments.reference --generate # zamrzni vrednosti obicnog minimaxa
python -m experiments.reference --verify   # replay: gate za faze 6-10
```

Svaka optimizacija u fazama 6-10 mora da vrati **istu vrednost** kao obican minimax.
`--verify` je ta provera; broj cvorova sme da se menja, vrednost ne sme.

## Pokretanje meca

```sh
python -m experiments.match --a greedy --b random --games 10000 --seed 2026
```

Rezultat je uvek razdvojen po tome ko je igrao prvi: sa nasumicnom igrom prvi igrac vec
dobija 55.4% partija (mereno u fazi 2), pa procenat pobeda bez te podele nista ne govori
o jacini agenta.

## Pokretanje testova

```sh
cd connect4
python3 -m venv .venv && source .venv/bin/activate
pip install pytest
pytest
```

## Konvencije

- Red 0 je **dno** table, kolona 0 je **levo**. Isti raspored kao bitboard
  (`bit = col * 7 + row`), da bi diferencijalno testiranje bilo poredjenje bez prevodjenja
  koordinata. Jedino GUI okrece redove, pri crtanju.
- Potez je indeks kolone, 0-6 interno.
- `from_move_string("4455")` koristi **1-bazirane** cifre, format standardnih benchmark
  skupova pozicija (faza 13).
