# Connect Four — agenti pretrage (diplomski rad)

Minimax / alfa-beta / negascout vs. MCTS na igri Connect Four.
Plan rada: [../plan/BUILD_PLAN.md](../plan/BUILD_PLAN.md)

## Struktura

```
connect4/
  core/          board_naive.py (referentna implementacija), rules.py (ugovor), types.py
  agents/        random, greedy, minimax, alphabeta, mcts, oracle        [faze 4-11]
  search/        evaluacija, uredjivanje poteza, transpoziciona tabela   [faze 5-10]
  gui/           pygame prikaz                                           [faza 3]
  experiments/   turniri, benchmark, ablacije, grafici                   [faze 12-14]
  data/          test pozicije, sirovi rezultati, slike
  tests/         pytest
```

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
