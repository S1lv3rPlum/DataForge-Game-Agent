# tests/test_agents.py
# Tests for the agent logic (no display, no pygame)
# Run with: pytest tests/test_agents.py -v

import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.learning_agent import DQNAgent, ReplayMemory

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = MinesweeperEnv(render_mode=None, difficulty="Beginner")
    e.reset()
    return e

@pytest.fixture
def agent(env):
    state_size  = env.rows * env.cols
    action_size = env.action_space.n
    return DQNAgent(state_size, action_size, difficulty="Beginner")

# ── Replay Memory Tests ───────────────────────────────────────────────────────

class TestReplayMemory:

    def test_memory_stores_experience(self):
        mem = ReplayMemory(100)
        mem.push([0]*81, 0, 1.0, [0]*81, False)
        assert len(mem) == 1

    def test_memory_respects_capacity(self):
        mem = ReplayMemory(10)
        for i in range(20):
            mem.push([0]*81, i, 1.0, [0]*81, False)
        assert len(mem) == 10

    def test_memory_samples_correct_size(self):
        mem = ReplayMemory(100)
        for i in range(50):
            mem.push([0]*81, i % 10, float(i), [0]*81, False)
        batch = mem.sample(16)
        assert len(batch) == 16

    def test_memory_cannot_sample_more_than_stored(self):
        mem = ReplayMemory(100)
        mem.push([0]*81, 0, 1.0, [0]*81, False)
        with pytest.raises(Exception):
            mem.sample(10)

# ── DQN Agent Tests ───────────────────────────────────────────────────────────

class TestDQNAgent:

    def test_agent_selects_valid_action(self, agent, env):
        obs, _ = env.reset()
        action = agent.select_action(obs, env)
        assert 0 <= action < env.action_space.n

    def test_epsilon_starts_at_one(self, agent):
        assert agent.epsilon == pytest.approx(1.0)

    def test_epsilon_decays(self, agent):
        initial = agent.epsilon
        agent.update_epsilon()
        assert agent.epsilon < initial

    def test_epsilon_never_below_min(self, agent):
        for _ in range(10000):
            agent.update_epsilon()
        assert agent.epsilon >= 0.05

    def test_learn_requires_enough_memory(self, agent, env):
        """Agent should not crash if memory too small to learn."""
        obs, _ = env.reset()
        # Only one experience — should not crash
        agent.memory.push(obs, 0, 1.0, obs, False)
        agent.learn()   # should silently skip

    def test_learn_runs_with_full_batch(self, agent, env):
        """Agent should learn without crashing with full batch."""
        obs, _ = env.reset()
        for i in range(100):
            action           = env.action_space.sample()
            next_obs, r, done, _, _ = env.step(action)
            agent.memory.push(obs, action, r, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        agent.learn()   # should not crash

    def test_action_masking_avoids_revealed(self, agent, env):
        """Agent should never try to reveal an already revealed cell."""
        obs, _ = env.reset()

        # Reveal some safe cells manually
        import numpy as np
        safe = np.argwhere(~env.mine_grid)
        for row, col in safe[:5]:
            env.step(row * env.cols + col)

        obs = env.visible.copy()

        # Run many action selections — none should reveal revealed cells
        for _ in range(200):
            action   = agent.select_action(obs, env)
            n        = env.rows * env.cols
            cell_idx = action % n
            row      = cell_idx // env.cols
            col      = cell_idx  % env.cols
            # If it's a reveal action, cell must be hidden
            if action < n:
                assert env.visible[row, col] == -1, \
                    f"Agent tried to reveal already-revealed cell ({row},{col})"

    def test_target_network_update(self, agent):
        """Target network update should not crash."""
        agent.update_target_network()

    def test_difficulty_saved_in_filename(self, agent):
        filename = agent._model_path()
        assert "beginner" in filename.lower()
