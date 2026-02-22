# random_agent.py
# Random Agent — Version 2.0
# Now with:
#   - Difficulty selection screen
#   - Flagging actions in random move set
#   - Updated dashboard metrics (flags, board cleared %)
#   - Shared dashboard support for comparison with learning agent
#
# Run this with:  python games/minesweeper/random_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pygame
from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.dashboard import Dashboard

# ── Settings ──────────────────────────────────────────────────────────────────
NUM_EPISODES = 100      # how many games to play
STEP_DELAY   = 0.05    # seconds between moves


def run():
    env  = MinesweeperEnv(render_mode="human")
    obs, _ = env.reset()   # triggers difficulty selection screen

    dash = Dashboard(agent_name="Random Agent")

    print(f"\nRandom Agent starting on {env.difficulty}...")
    print(f"Board: {env.rows}x{env.cols} | Mines: {env.num_mines}")
    print(f"Action space: {env.action_space.n} "
          f"({env.rows * env.cols} reveal + "
          f"{env.rows * env.cols} flag)\n")

    for episode in range(1, NUM_EPISODES + 1):
        if episode > 1:
            obs, _ = env.reset()

        total_reward    = 0
        done            = False
        step            = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    dash.close()
                    return

            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step         += 1
            time.sleep(STEP_DELAY)

        result = "WIN 🎉" if env.won else "LOSS 💥"
        print(f"Episode {episode:4d} | {result} | "
              f"Steps: {step:3d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Flags: {env.correct_flags}/{env.num_mines} | "
              f"Cleared: {env.safe_revealed}/{env.safe_total}")

        dash.update(
            episode       = episode,
            reward        = total_reward,
            won           = env.won,
            steps         = step,
            correct_flags = env.correct_flags,
            pct_cleared   = env.safe_revealed / env.safe_total
                            if env.safe_total > 0 else 0.0
        )

        time.sleep(0.8)

    env.close()
    dash.close()


if __name__ == "__main__":
    run()
