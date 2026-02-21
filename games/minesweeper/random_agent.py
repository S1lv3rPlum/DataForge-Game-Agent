# random_agent.py
# A completely random agent — picks cells with no strategy whatsoever
# Now with live performance dashboard!
#
# Run this with:  python games/minesweeper/random_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pygame
from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.dashboard import Dashboard

# ── Settings ────────────────────────────────────────────────────────────────
NUM_EPISODES = 50       # how many games to play
STEP_DELAY   = 0.15     # seconds between moves (slow enough to watch)
PRINT_BOARD  = False    # set True if you want terminal board output too


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
    env  = MinesweeperEnv(render_mode="human")
    dash = Dashboard(agent_name="Random Agent", color="#4C72B0")

    for episode in range(1, NUM_EPISODES + 1):
        obs, _       = env.reset()
        total_reward = 0
        done         = False
        step         = 0

        print(f"\nEpisode {episode}/{NUM_EPISODES}")

        while not done:
            # Handle pygame window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    dash.close()
                    print("\nWindow closed. Exiting.")
                    return

            # Random action — no strategy whatsoever
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step         += 1

            if PRINT_BOARD:
                print(f"Step {step} — Action: row={action // env.cols}, "
                      f"col={action % env.cols} — Reward: {reward:.1f}")
                print_board(obs)

            time.sleep(STEP_DELAY)

        # Episode complete — update dashboard
        result = "WIN 🎉" if env.won else "LOSS 💥"
        print(f"Episode {episode} — {result} | "
              f"Steps: {step} | Reward: {total_reward:.1f}")

        dash.update(
            episode = episode,
            reward  = total_reward,
            won     = env.won,
            steps   = step
        )

        time.sleep(1)

    env.close()
    dash.close()   # keeps dashboard open after run finishes


if __name__ == "__main__":
    run()
