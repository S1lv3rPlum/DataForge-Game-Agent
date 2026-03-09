# tests/test_integration.py
# Integration tests — full episode runs headless
# Run with: pytest tests/test_integration.py -v

import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.learning_agent import DQNAgent
from agent.dashboard import Dashboard, SHARED_FILE

# ── Full Episode Tests ────────────────────────────────────────────────────────

class TestFullEpisode:

    def test_random_agent_completes_episode(self):
        """Random agent should always complete an episode."""
        env    = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        obs, _ = env.reset()
        done   = False
        steps  = 0

        while not done and steps < 1000:
            action            = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            steps            += 1

        assert done == True
        assert steps < 1000
        env.close()

    def test_learning_agent_completes_episode(self):
        """Learning agent should always complete an episode."""
        env         = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        obs, _      = env.reset()
        state_size  = env.rows * env.cols
        action_size = env.action_space.n
        agent       = DQNAgent(state_size, action_size,
                               difficulty="Beginner")
        done  = False
        steps = 0

        while not done and steps < 1000:
            action             = agent.select_action(obs, env)
            obs, _, done, _, _ = env.step(action)
            steps             += 1

        assert done == True
        env.close()

    def test_ten_random_episodes_complete(self):
        """Run 10 random episodes without error."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")

        for ep in range(10):
            obs, _ = env.reset()
            done   = False
            steps  = 0

            while not done and steps < 500:
                action             = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                steps             += 1

            assert done == True

        env.close()

    def test_dashboard_tracks_ten_episodes(self):
        """Dashboard should correctly track 10 episodes."""
        env  = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        dash = Dashboard(agent_name="Random Agent", show_window=False)

        for ep in range(10):
            obs, _ = env.reset()
            done   = False
            steps  = 0

            while not done and steps < 500:
                action             = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                steps             += 1

            dash.update(
                episode       = ep + 1,
                reward        = 1.0 if env.won else -1.0,
                won           = env.won,
                steps         = steps,
                correct_flags = env.correct_flags,
                pct_cleared   = env.safe_revealed / env.safe_total
            )

        total = dash.wins + dash.losses
        assert total == 10
        assert len(dash.steps_history) == 10

        env.close()
        dash.close()

        if os.path.exists(SHARED_FILE):
            os.remove(SHARED_FILE)

    def test_all_difficulties_complete_episode(self):
        """Each difficulty should complete at least one episode."""
        for diff in ["Beginner", "Medium", "Hard"]:
            env    = MinesweeperEnv(render_mode=None, difficulty=diff)
            obs, _ = env.reset()
            done   = False
            steps  = 0

            while not done and steps < 5000:
                action             = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                steps             += 1

            assert done == True, f"{diff} episode never completed"
            env.close()
```

---

Now add pytest to `requirements.txt`. Open it and add this line:
```
pytest>=7.0
```

Then install it on Matt's PC once with:
```
pip install pytest
```

And run all tests with:
```
pytest tests/ -v
```

Or run a single file with:
```
pytest tests/test_minesweeper_env.py -v
