# Connect Four — agenti pretrage (diplomski rad)

Minimax / alfa-beta / negascout vs. MCTS na igri Connect Four.
Plan rada: [../plan/BUILD_PLAN.md](../plan/BUILD_PLAN.md)

## Pokretanje aplikacije

```sh
cd connect4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Nema nijednog argumenta komandne linije — sve se bira u prozoru:

- **režim igre**: čovek vs mašina, mašina vs mašina, čovek vs čovek;
- **algoritam po sedištu**: Random, Greedy, Minimax, Negamax (kasnije NegaScout i MCTS);
- **optimizacije po sedištu**: alfa-beta odsecanje, sortiranje poteza (killer potezi +
  istorijska heuristika), tabela transpozicija i progresivno produbljivanje, i svaka
  komponenta funkcije procene (pretnje, centar, parnost) nezavisno;
- **dubina pretrage**, sa izmerenom cenom najgoreg poteza uz klizač — ili, kada je
  progresivno produbljivanje uključeno, **vreme po potezu**, pa dubina postaje rezultat.

Ispod svakog sedišta stoji šta je poslednja pretraga koštala: čvorova, odsecanja, kvalitet
redosleda poteza, dubina, vrednost i vreme, plus zbir za celu partiju. To je ono što čini
prekidače čitljivim — ista pozicija, ista dubina, isti algoritam, samo jedno čekirano
polje razlike:

| sedište | agent | čvorova | odsecanja | vreme |
|---|---|---|---|---|
| alfa-beta uključena | `alphabeta-d6` | 3.425 | 820 | 0,063 s |
| alfa-beta isključena | `negamax-d6` | 136.609 | 0 | 3,000 s |

**Dubina je ograničena na ono što je izmereno.** Tabele `UNPRUNED_WORST_SECONDS`,
`PRUNED_WORST_SECONDS`, `ORDERED_WORST_SECONDS` i `TABLE_WORST_SECONDS` u `session.py`
drže najgore izmereno vreme jednog poteza po dubini, jednu tabelu po nivou optimizacije, a
granica klizača je dužina te tabele. Zato je **granica dubine sama po sebi rezultat**: 7
bez odsecanja, 9 sa odsecanjem, 11 sa odsecanjem i sortiranjem poteza, 13 sa tabelom
transpozicija.

**Progresivno produbljivanje ukida tu lestvicu umesto da je produži.** Pretraga fiksne
dubine nema odgovor dok se ne završi, pa se jedino može ograničiti dubina i izmeriti šta
to košta. Pretraga koja može da se prekine ograničena je svojim budužetom: klizač dubine
se tada gaši, granica se pušta, a panel pokazuje **dostignutu** dubinu i broj iteracija.
To je jedini prekidač u prozoru koji najgori slučaj čini *manjim* — sa 34 s (dubina 13) na
onoliko sekundi koliko je izabrano, uz izmereno prekoračenje od oko 1 ms. Pretraga se
izvršava u zasebnoj niti, pa prozor i tada crta 60 slika u sekundi i može da se zatvori.

## Struktura

```
connect4/
  main.py        ulazna tacka, bez argumenata                                             [3]
  config.py      geometrija, boje, fontovi (ne uvozi pygame na vrhu)                      [3]
  session.py     tok partije BEZ pygamea: sedista, potezi, kraj, izbor agenta             [3]
  game.py        pygame petlja, kontrolni panel, radna nit za pretragu                    [3]
  core/          rules.py (ugovor), types.py (+ Score skala), board_naive.py, bitboard.py [1-2, 5]
  agents/        base.py (interfejs), random_agent.py, greedy_agent.py                    [4]
                 minimax.py (MAX/MIN, ogranicen dubinom)                                  [5]
                 alphabeta.py (negamax; alfa-beta koja se iskljucuje)                     [6]
                 negascout, mcts, oracle                                                  [10-11, 13]
  search/        stats.py (Budget + SearchStats)                                          [4]
                 ordering.py (centar-prvo [4]; killer potezi + istorija, ukljucivi)       [7]
                 evaluation.py (pretnje, centar, parnost -- svaka iskljuciva)             [5]
                 transposition.py (position+mask kljuc, EXACT/LOWER/UPPER, dict i niz)    [8]
                 deepening.py (progresivno produbljivanje + prekid po vremenu/cvorovima)  [9]
  gui/           renderer.py (crtanje), widgets.py (kontrole), sprites.py (diskovi)       [3]
  experiments/   match.py (mec izmedju dva agenta, sa semenom)                            [4]
                 reference.py (zamrznute vrednosti), scaling.py (cvorovi i kriva)         [5-6]
                 turniri, benchmark, ablacije, grafici                                    [12-14]
  data/          testsets/ (referentne vrednosti), results/ (CSV), figures/
  tests/         pytest
```

**Prozor ne odlučuje ništa.** `session.py` zna šta sme da se odigra i čiji je potez;
`gui/` samo crta i pretvara klik u broj kolone. Zato ceo tok partije ima testove koji ne
traže pygame, a `experiments/` ostaje merodavan: nijedan broj u radu ne dolazi iz prozora.


## Merenja

```sh
python -m experiments.scaling --nodes      # cvorovi po dubini, obican minimax
python -m experiments.scaling --curve      # rezultat protiv greedyja po dubini
python -m experiments.scaling --pruning    # alfa-beta po dubini, prazna tabla
python -m experiments.scaling --positions  # isti odnos nad 200 realnih pozicija
python -m experiments.scaling --ordering        # redosled poteza: prazna tabla + 200 pozicija
python -m experiments.scaling --ordering-games  # isto, ali kroz cele partije
python -m experiments.scaling --density         # koliko stabla su uopste ponovljene pozicije
python -m experiments.scaling --table           # tabela transpozicija nad 200 pozicija
python -m experiments.scaling --table-empty     # isto, prazna tabla, do dubine 12
python -m experiments.scaling --table-games     # isto, kroz cele partije (+ trajna tabela)
python -m experiments.scaling --deepening        # progresivno produbljivanje nad 200 pozicija
python -m experiments.scaling --deepening-empty  # isto, prazna tabla, do dubine 12
python -m experiments.scaling --deepening-games  # isto, kroz cele partije
python -m experiments.scaling --deepening-clock  # da li se vremenski budzet postuje
python -m experiments.reference --generate # zamrzni vrednosti obicnog minimaxa
python -m experiments.reference --verify --agent alphabeta   # gate za faze 6-10
python -m experiments.reference --verify --agent alphabeta --ordering center_first
python -m experiments.reference --verify --agent alphabeta --table no_table
python -m experiments.reference --verify --agent alphabeta --deepening iterative+pv+order
```

Svaka optimizacija u fazama 6-10 mora da vrati **istu vrednost** kao obican minimax.
`--verify` je ta provera; broj cvorova sme da se menja, vrednost ne sme. `--agent negamax`
poredi i broj cvorova, jer taj rezim nema odsecanja pa mu je stablo isto kao minimaxu.
Sve cetiri imenovane strategije redosleda daju 0 neslaganja nad svih 1.050 zapisa.

**Od faze 6 nadalje broj cvorova nije uporediv bez funkcije procene i redosleda poteza.**
Obican minimax nema odsecanja, pa ga ni jedno ni drugo ne moze skrenuti; alfa-beta odseca
na osnovu onoga sto je vec nasla, pa zavisi od oba. Od faze 8 tome se dodaje i tabela
transpozicija, a od faze 9 i **budzet** — pretraga zaustavljena satom nema ponovljiv broj
cvorova. Zato svaki CSV u `data/results/` nosi kolone `evaluation`, `ordering`,
`transposition` i `budget`, a svaki broj koji se citira u radu dolazi iz budzeta po dubini
ili po cvorovima; vreme se koristi samo tamo gde se vreme i meri.

## Pokretanje meca

```sh
python -m experiments.match --a greedy --b random --games 10000 --seed 2026
python -m experiments.match --a alphabeta --b greedy --games 200 --opening-plies 4
```

Rezultat je uvek razdvojen po tome ko je igrao prvi: sa nasumicnom igrom prvi igrac vec
dobija 55.4% partija (mereno u fazi 2), pa procenat pobeda bez te podele nista ne govori
o jacini agenta. Deterministicki agenti bez `--opening-plies` igraju uvek istu partiju,
na sta izvestaj upozorava.

## Pokretanje testova

```sh
cd connect4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Testovi prozora se **preskaču** ako pygame nije instaliran, da bi merni deo projekta
mogao da se pokrene i na interpreteru koji ga nema (npr. PyPy, BUILD_PLAN 2.3).

## Konvencije

- Red 0 je **dno** table, kolona 0 je **levo**. Isti raspored kao bitboard
  (`bit = col * 7 + row`), da bi diferencijalno testiranje bilo poredjenje bez prevodjenja
  koordinata. Jedino GUI okrece redove, pri crtanju.
- Potez je indeks kolone, 0-6 interno.
- `from_move_string("4455")` koristi **1-bazirane** cifre, format standardnih benchmark
  skupova pozicija (faza 13).
