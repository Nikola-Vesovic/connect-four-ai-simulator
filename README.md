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
- **algoritam po sedištu**: Random, Greedy, Minimax, Negamax, MCTS;
- **optimizacije po sedištu**: alfa-beta odsecanje, sortiranje poteza (killer potezi +
  istorijska heuristika), tabela transpozicija, progresivno produbljivanje i negascout, i
  svaka komponenta funkcije procene (pretnje, centar, parnost) nezavisno;
- **za MCTS, pet sopstvenih prekidača na istim tim redovima**: greedy simulacije, čuvanje
  stabla, dokazivanje ishoda, vremensko ograničenje i konstanta istraživanja C;
- **dubina pretrage**, sa izmerenom cenom najgoreg poteza uz klizač — ili, kada je
  progresivno produbljivanje uključeno, **vreme po potezu**, pa dubina postaje rezultat;
  za MCTS na tom mestu stoji **broj simulacija**.

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

**NegaScout je jedina optimizacija koja lestvicu niti produžava niti ukida.** Ona pretragu
samo pojeftinjuje, pa se konfiguracija sa njom računa po istoj prečki kao i bez nje —
procena tako ostaje konzervativna. Prva moće da bude i skuplja: nad 200 realnih pozicija
ona gubi na dubini 1 (0,888x) i 2 (0,912x), izjednačuje se na 3, a dobitak počinje od
dubine 4 i raste do **1,303x na dubini 7**. Panel zato prikazuje i **ponovne pretrage** —
udeo pretraga nultim prozorom koje su morale da se ponove, jedini broj u panelu kod koga
je manje bolje.

**MCTS nije šesto polje za čekiranje nego peti algoritam, i to je pravilo a ne ukus.**
Pet optimizacija iznad ne mogu da promene odigrani potez — kupuju isti odgovor za manje
posla, zbog čega alfa-beta i negascout nisu zasebni algoritmi nego prekidači. MCTS pada
na tom testu u svakom pravcu: ne računa minimax vrednost, ne poziva funkciju procene i
namerno igra druge poteze. Zato dobija svoje mesto u biraču, sopstvene prekidače na istih
pet redova panela (prozor ne raste) i **sopstvene statistike**: simulacije, čvorove,
dubinu stabla, procenu pobede, koncentraciju i raspodelu poseta po sedam kolona.

**Jedna simulacija nije jedan čvor, i to je najvažnija merna odluka faze 11.** Simulacija
prođe put kroz stablo, doda jedan čvor i odigra celu partiju do kraja. Mereno nad istih
25 pozicija: greedy simulacija košta **214,3 pozicije**, nasumična **15,1** — 14,2 puta
manje. Zato se svako poređenje prijavljuje dvaput, pri jednakom broju simulacija i pri
jednakom vremenu, i uvek se kaže koja je jedinica u pitanju.

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
                 mcts.py (UCT: izbor, sirenje, simulacija, propagacija sa obrtanjem)      [11]
                 oracle.py (ogranicen savrseni resavac, samo za merenje)                  [13]
  search/        stats.py (Budget + SearchStats)                                          [4]
                 ordering.py (centar-prvo [4]; killer potezi + istorija, ukljucivi)       [7]
                 evaluation.py (pretnje, centar, parnost -- svaka iskljuciva)             [5]
                 transposition.py (position+mask kljuc, EXACT/LOWER/UPPER, dict i niz)    [8]
                 deepening.py (progresivno produbljivanje + prekid po vremenu/cvorovima)  [9]
                 scouting.py (negascout: nulti prozor + ponovna pretraga)                 [10]
                 playout.py (UCT: politika simulacije, C, cuvanje stabla, dokazivanje)    [11]
  gui/           renderer.py (crtanje), widgets.py (kontrole), sprites.py (diskovi)       [3]
  experiments/   match.py (mec izmedju dva agenta, sa semenom)                            [4]
                 reference.py (zamrznute vrednosti), scaling.py (cvorovi i kriva)         [5-6]
                 tournament.py (round-robin, deonice, Elo sa intervalima)                 [12]
                 runner.py (turnir iz JSON konfiguracije, preko vise jezgara)             [12]
                 benchmark.py (poredjenje sa savrsenim resavacem, preko vise jezgara)     [13]
                 ablacije, grafici                                                        [14]
  data/          testsets/ (referentne vrednosti), configs/ (turniri), results/ (CSV), figures/
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
python -m experiments.scaling --scout            # negascout nad 200 pozicija
python -m experiments.scaling --scout-empty      # isto, prazna tabla, do dubine 12
python -m experiments.scaling --scout-games      # isto, kroz cele partije
python -m experiments.scaling --scout-ordering   # slajd 148: negascout protiv 4 redosleda
python -m experiments.scaling --scout-deepening  # negascout sa i bez produbljivanja
python -m experiments.scaling --mcts-cost         # koliko kosta jedna simulacija
python -m experiments.scaling --mcts-tactics      # mat u jedan i odbrana od njega, po budzetu
python -m experiments.scaling --mcts-convergence  # da li posete ostaju na dobitnom potezu
python -m experiments.scaling --mcts-ablation     # svi prekidaci, pri jednakom broju simulacija
python -m experiments.scaling --mcts-rollout-time # ista politika simulacije, ali pri jednakom vremenu
python -m experiments.scaling --mcts-vs-alphabeta # obecano poredjenje: jednako vreme I jednaki cvorovi
python -m experiments.reference --generate # zamrzni vrednosti obicnog minimaxa
python -m experiments.reference --verify --agent alphabeta   # gate za faze 6-10
python -m experiments.reference --verify --agent alphabeta --ordering center_first
python -m experiments.reference --verify --agent alphabeta --table no_table
python -m experiments.reference --verify --agent alphabeta --deepening iterative+pv+order
python -m experiments.reference --verify --agent alphabeta --scout no_scout
```

Svaka optimizacija u fazama 6-10 mora da vrati **istu vrednost** kao obican minimax.
`--verify` je ta provera; broj cvorova sme da se menja, vrednost ne sme. `--agent negamax`
poredi i broj cvorova, jer taj rezim nema odsecanja pa mu je stablo isto kao minimaxu.
Sve cetiri imenovane strategije redosleda daju 0 neslaganja nad svih 1.050 zapisa.

**Za MCTS ta provera ne postoji i to treba reci naglas.** On ne racuna minimax vrednost,
pa `reference.py` nad njim nema sta da uporedi — i strukturno ne moze, jer tog agenta
nikada ni ne pravi. Umesto jedne jednakosti stoje **svojstva**, podeljena na dokaziva
(isto seme + isti budzet daje isti potez i iste brojace; stablo ostaje u ravnotezi;
zavrsena pozicija se nikada ne simulira; uz `dokazivanje ishoda` mat u jedan se **ne moze**
propustiti) i **statisticka** (koliko cesto obican UCT nadje mat u jedan, koliko cesto ga
odbrani, koliko cesto posete ostanu na potezu koji cuva dobijenu poziciju) — a statisticka
se prijavljuju kao stope nad uzorkom cija je velicina navedena.

**Od faze 6 nadalje broj cvorova nije uporediv bez funkcije procene i redosleda poteza.**
Obican minimax nema odsecanja, pa ga ni jedno ni drugo ne moze skrenuti; alfa-beta odseca
na osnovu onoga sto je vec nasla, pa zavisi od oba. Od faze 8 tome se dodaje i tabela
transpozicija, a od faze 9 i **budzet** — pretraga zaustavljena satom nema ponovljiv broj
cvorova, a od faze 10 i **varijanta pretrage** (negascout ili ne). Zato svaki CSV u
`data/results/` nosi kolone `evaluation`, `ordering`,
`transposition` i `budget`, a svaki broj koji se citira u radu dolazi iz budzeta po dubini
ili po cvorovima; vreme se koristi samo tamo gde se vreme i meri.

## Pokretanje meca

```sh
python -m experiments.match --a greedy --b random --games 10000 --seed 2026
python -m experiments.match --a alphabeta --b greedy --games 200 --opening-plies 4
python -m experiments.match --a mcts --b mcts-random --games 120 --opening-plies 4
```

Rezultat je uvek razdvojen po tome ko je igrao prvi: sa nasumicnom igrom prvi igrac vec
dobija 55.4% partija (mereno u fazi 2), pa procenat pobeda bez te podele nista ne govori
o jacini agenta. Deterministicki agenti bez `--opening-plies` igraju uvek istu partiju,
na sta izvestaj upozorava.

## Pokretanje turnira

```sh
python -m experiments.runner --config data/configs/ladder.json --workers 8
python -m experiments.runner --config data/configs/ladder.json --check 8      # gate
python -m experiments.runner --config data/configs/speedup.json --speedup 1,2,4,6,8 \
       --granularity 1,2,4,16                                                # ubrzanje
```

Turnir se opisuje **JSON datotekom** u `data/configs/`: agenti, broj partija, seme,
nasumicni otvor i budzet. Jedna komanda igra round-robin i pise dva CSV-a u
`data/results/` — jedan red po paru i jedan red po agentu sa Elo ocenom.

- **Partije se dele po jezgrima.** Partija je ciste funkcija svog rednog broja (seme para
  je odredjeno glavnim semenom, a `new_game` brise sve sto pretrazivanje pamti izmedju
  partija), pa deonica nije priblizna nego *ista* igra. Isti otisak se dobija i sa jednim
  procesom i sa osam, i pri svakoj granulaciji deonica.
- **Vremenski budzet se u paraleli ODBIJA, ne upozorava.** Mec pod satom pod opterecenjem
  meri opterecenje: faza 9 je istih 540 partija pod satom ocenila na 44,3%, a pod budzetom
  po cvorovima na 48,5%, i samo je drugo bio kontrolisan ogled. Uz `--workers 1` je
  dozvoljen, a izvestaj tada kaze da se ne ponavlja.
- **Elo nije procenat pobeda.** Ocene se dobijaju zajednickim fitom nad celim turnirom
  (Bradley-Terry, remi vredi pola poena), sa malim priorom bez kojeg agent koji nije
  osvojio nijedan poen nema konacnu ocenu. Interval poverenja je butstrep: isti turnir se
  1.000 puta ponovo izvlaci iz izmerenih ucestalosti i fit se ponavlja.

## Poredjenje sa savrsenim resavacem (faza 13)

```sh
python -m experiments.benchmark --verify --positions 1000,500 --workers 8 --write
python -m experiments.benchmark --score  --positions 1000,200 --workers 8 --write
python -m experiments.benchmark --check 8 --sets end-easy --positions 24     # gate
```

Ovo je jedino merenje u radu koje agenta ne poredi sa drugim agentom nego sa **tacnom
vrednoscu pozicije**. Tece u tri koraka i prvi se ne preskace:

1. **Dokazi resavac.** Nad objavljenim pozicijama Pascala Ponsa
   (`data/testsets/gamesolver/`) nasa pretraga puno dubine reprodukuje tacan rezultat — i
   ishod i udaljenost do mata. Izmereno: **1.551 pozicija, 100%, bez ijednog neslaganja.**
   Provera ide u oba smera: skup potvrdjuje nas resavac, a nas resavac potvrdjuje skup.
2. **Proceni svako dete.** Datoteka daje vrednost *pozicije*, ne pojedinacnog poteza, pa se
   vrednost svakog legalnog poteza racuna posebno i iz njih se dobija skup poteza koji
   cuvaju vrednost.
3. **Oceni agente:** procenat optimalnih poteza, procenat grubih gresaka i prosecan gubitak.

- **Tri mere nisu jedna te ista.** *Optimalan* znaci "u skupu koji cuva vrednost" — takvih
  poteza u proseku ima 1,38, pa se nikada ne kaze "najbolji potez". *Gruba greska* menja
  teorijski **ishod**; pobeda u 11 poteza umesto u 5 jeste losiji potez ali nije gruba
  greska. *Gubitak vrednosti* je u polupotezima unutar istog ishoda, a oko 10.000 kada se
  ishod promeni, pa se prikazuje razlozeno.
- **Dve populacije se prikazuju odvojeno.** Pozicija u kojoj svaki potez cuva vrednost ne
  razdvaja nikoga (**23,5% skupa End-Easy** je takvo), a u izgubljenoj poziciji gruba greska
  nije ni moguca, pa je 0% tamo svojstvo pozicija a ne jacine agenta.
- **Dokaz je jeftiniji od potpunog resavanja.** Vrednost u pobednickom ili gubitnickom opsegu
  je tacna bez obzira na to na kojoj se dubini pretraga zaustavila; samo remi trazi celo
  stablo. Zato resavac radi progresivno i sa budzetom u cvorovima, a pozicija koju ne uspe da
  dokaze se **izbacuje i prebroji**, nikada ne pogadja.

## Ablaciona matrica i figure (faza 14)

Ova faza nista ne meri iznova osim same matrice — sve ostalo cita vec zamrznute CSV
fajlove iz `data/results/`.

```sh
python -m experiments.ablation --matrix    # svih 8 kombinacija O/T/S, ~48 s
python -m experiments.ablation --control   # sinteticka kontrola za paralelizaciju

pip install -r requirements-figures.txt    # matplotlib NIJE potreban za igru
python -m experiments.plots --all --both-languages   # 6 figura, pdf + png, latinica i cirilica
python -m experiments.plots --check        # dokaz da se ponovnim pokretanjem dobijaju isti bajtovi
python -m experiments.plots --glyphs       # dokaz da se sva slova zaista iscrtavaju
```

- **Matrica prikazuje KOMBINACIJE, ne spisak procenata.** Pored izmerenog dobitka stoji
  **proizvod** pojedinacnih dobitaka, pa se preklapanje vidi kao broj. Sortiranje poteza se
  preklapa sa svim ostalim; jedini izuzetak u porodici je par *tabela + negascout*, koji daje
  **vise** od proizvoda (109,8% / 107,3% / 112,3% na dubinama 4/5/6), jer cvor pretrazen
  nultim prozorom ne moze da upise tacnu vrednost, pa zapis iz tabele prestaje da *odgovara*
  na cvor i pocinje da *nudi potez*.
- **Dve kombinacije nisu novi brojevi**: osnovna reprodukuje `alphabeta_by_position.csv`, a
  puna `scout_by_position.csv`, u cvor — i to se proverava dok merenje traje.
- **Svaki broj sa figure postoji i u tabeli.** `data/figures/RESULTS.md` se generise iz istih
  objekata iz kojih se crtaju figure, pa se ne mogu razici.
- `requirements.txt` ostaje `pygame` + `pytest`. Aplikacija i testovi rade i bez matplotliba.

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
