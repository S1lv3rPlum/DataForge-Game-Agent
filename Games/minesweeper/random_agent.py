# random_agent.py
# A completely random agent — picks cells with no strategy whatsoever
# This is Step 1: prove the environment works before adding any learning
#
# Run this with:  python games/minesweeper/random_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pygame
from games.minesweeper.minesweeper_env import MinesweeperEnv

# ── Settings ────────────────────────────────────────────────────────────────
NUM_EPISODES    = 10       # how many games to play
STEP_DELAY      = 0.3      # seconds between moves (slow enough to watch)
PRINT_BOARD     = True     # print board state to terminal as well


def print_board(obs):
    """Print a readable version of the board to the terminal."""
    symbols = {-1: "■", -2: "F", -3: "💣"}
    print("\n  " + " ".join(str(c) for c in range(obs.shape[1])))
    for r, row in enumerate(obs):
        line = f"{r} "
        for val in row:
            if val in symbols:
                line += symbols[val] + " "
            else:
                line += str(val) + " "
        print(line)
    print()


def run():
    env = MinesweeperEnv(render_mode="human")

    wins   = 0
    losses = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(f"\n{'='*40}")
        print(f"Episode {episode} of {NUM_EPISODES}")
        print(f"{'='*40}")

        while not done:
            # Handle pygame window close button
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    print("\nWindow closed. Exiting.")
                    return

            # Pick a completely random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            if PRINT_BOARD:
                print(f"Step {step} — Action: row={action // env.cols}, "
                      f"col={action % env.cols} — Reward: {reward:.1f}")
                print_board(obs)

            time.sleep(STEP_DELAY)

        # Episode summary
        result = "🎉 WIN" if env.won else "💥 LOSS"
        if env.won:
            wins += 1
        else:
            losses += 1

        print(f"\nEpisode {episode} complete — {result}")
        print(f"Steps: {step} | Total Reward: {total_reward:.1f}")
        print(f"Record so far — Wins: {wins} | Losses: {losses}")

        # Pause between episodes so you can see the final board
        time.sleep(2)

    env.close()
    print(f"\n{'='*40}")
    print(f"All episodes complete!")
    print(f"Final Record — Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {wins/NUM_EPISODES*100:.1f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    run()
