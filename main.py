import argparse

from contexto_solver.pipeline import play_game_and_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a single Contexto game")
    parser.add_argument("--game-id", type=int, default=None, help="Game ID to play (default: next after current)")
    args = parser.parse_args()
    play_game_and_record(game_id=args.game_id)
