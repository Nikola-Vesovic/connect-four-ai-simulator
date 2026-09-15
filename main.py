"""Entry point. `python main.py` -- no arguments, nothing to configure on the command line.

Everything is chosen in the window: who plays each seat, which algorithm, how deep, and
which optimisations are switched on. That is deliberate. The application has to run on
someone else's machine without instructions.

    python -m pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import traceback

import pygame

from game import Game


def main() -> None:
    try:
        pygame.init()
        Game().run()
    except Exception:
        traceback.print_exc()
        input("Došlo je do greške. Pritisni Enter za izlaz...")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
