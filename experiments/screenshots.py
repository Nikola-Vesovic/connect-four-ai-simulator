"""Capture the README screenshots through the application's real session and renderer."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from session import Controller, SeatSpec, Session

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "figures"
HUMAN_GAME = (7, 6, 6, 3, 4, 3, 7, 2, 7, 2, 2, 6, 6, 5, 1, 7, 2, 6, 5, 7, 7, 3, 2, 4, 4, 6, 3)


def make_sessions() -> dict[str, Session]:
    agents = Session((
        SeatSpec(controller=Controller.MACHINE, algorithm="negamax", depth=6),
        SeatSpec(controller=Controller.MACHINE, algorithm="mcts", simulations=1000),
    ))
    while agents.board.moves_played < 18 and not agents.is_over:
        move, stats = agents.plan()
        agents.commit(move, stats)

    humans = Session((SeatSpec(), SeatSpec()))
    for column in HUMAN_GAME:
        assert humans.drop(column - 1)
    assert humans.is_over and humans.winning_cells()
    return {"simulator_agents": agents, "simulator_humans": humans}


def capture(output: Path = OUTPUT) -> dict:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    from game import Game

    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    pygame.init()
    try:
        for name, session in make_sessions().items():
            for seat in range(2):
                records = [record for record in session.history if record.seat == seat]
                assert session.totals[seat].moves == len(records)
                assert session.totals[seat].nodes == sum(
                    record.stats.nodes for record in records if record.stats is not None
                )
            game = Game(session)
            game.paused = not session.is_over
            game.draw()
            width, height = game.screen.get_size()
            colors = {
                tuple(game.screen.get_at((pixel_x, pixel_y)))
                for pixel_y in range(0, height, 16)
                for pixel_x in range(0, width, 16)
            }
            assert len(colors) > 12, "Screenshot is unexpectedly blank."
            pygame.image.save(game.screen, str(output / f"{name}.png"))
            metadata[name] = {
                "seed": session.seed,
                "size": [width, height],
                "moves": [record.column + 1 for record in session.history],
                "result": session.result.name,
                "seats": [asdict(spec) for spec in session.seats],
                "totals": [asdict(total) for total in session.totals],
                "searches": [
                    {
                        "seat": record.seat,
                        "column": record.column + 1,
                        "nodes": record.stats.nodes,
                        "simulations": record.stats.simulations,
                        "seconds": record.stats.elapsed,
                    }
                    for record in session.history if record.stats is not None
                ],
            }
        (output / "screenshots.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        pygame.quit()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for name, details in capture(args.output).items():
        print(f"Captured {name}.png: {len(details['moves'])} moves, {details['result']}")


if __name__ == "__main__":
    main()