# random_agent.py
# Random Agent — Version 3.0
# Now a reusable module launchable from the main launcher
# or directly from the terminal.
#
# Run directly:  python agent/random_agent.py
# Or via launcher: python launch.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pygame
from games.registry import GAME_REGISTRY
from agent.dashboard import Dashboard

# ── Settings ──────────────────────────────────────────────────────────────────
NUM_EPISODES = 100
STEP_DELAY   = 0.05


def run(game_name="Minesweeper"):
    cfg      = GAME_REGISTRY[game_name]
    EnvClass = cfg["env_class"]

    env    = EnvClass(render_mode="human")
    obs, _ = env.reset()

    dash = Dashboard(agent_name="Random Agent")

    print(f"\n🎲 Random Agent starting — {game_name}")
    if hasattr(env, "difficulty"):
        print(f"Difficulty : {env.difficulty}")
    if hasattr(env, "rows"):
        print(f"Board      : {env.rows}x{env.cols}")
        print(f"Mines      : {env.num_mines}")
    print(f"Episodes   : {NUM_EPISODES}\n")

    for episode in range(1, NUM_EPISODES + 1):
        if episode > 1:
            obs, _ = env.reset()

        total_reward = 0
        done         = False
        step         = 0

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

def run_headless(game_name="Minesweeper", dash=None,
                 stop_flag=None, watch_request=None):
    """
    Runs random agent — renders only when watch_request matches.
    Used by launcher threading architecture.
    """
    import time
    from games.registry import GAME_REGISTRY

    cfg      = GAME_REGISTRY[game_name]
    EnvClass = cfg["env_class"]

    if stop_flag is None:
        stop_flag = {"done": False}

    episode = 0
    TOTAL   = 3000

    for _ in range(TOTAL):
        if stop_flag.get("done"):
            break
        if watch_request and watch_request.get("agent") == "MENU":
            break

        watching = (watch_request and
                    watch_request.get("agent") == "Random Agent")

        env      = EnvClass(
            render_mode="human" if watching else None,
            difficulty=None
        )
        obs, _   = env.reset()
        done     = False
        reward   = 0
        steps    = 0

        while not done:
            action       = env.action_space.sample()
            obs, r, done, _, _ = env.step(action)
            reward      += r
            steps       += 1
            if watching:
                time.sleep(0.05)

        env.close()
        episode += 1

        if dash:
            dash.update(
                episode       = episode,
                reward        = reward,
                won           = env.won,
                steps         = steps,
                correct_flags = env.correct_flags,
                pct_cleared   = env.safe_revealed / env.safe_total
                                if env.safe_total > 0 else 0.0
            )

        print(f"[Random] Ep {episode:4d} | "
              f"{'WIN 🎉' if env.won else 'LOSS 💥'} | "
              f"Steps: {steps:3d} | Reward: {reward:6.1f}")

if __name__ == "__main__":
    run()
