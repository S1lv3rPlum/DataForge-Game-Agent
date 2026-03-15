# tests/test_minesweeper_env.py
# Tests for the Minesweeper environment
# Run with: pytest tests/test_minesweeper_env.py -v

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from games.minesweeper.minesweeper_env import MinesweeperEnv, HIDDEN, FLAGGED, MINE

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def beginner_env():
    """Create a headless Beginner environment."""
    env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
    env.reset()
    return env

@pytest.fixture
def medium_env():
    env = MinesweeperEnv(render_mode=None, difficulty="Medium")
    env.reset()
    return env

@pytest.fixture
def hard_env():
    env = MinesweeperEnv(render_mode=None, difficulty="Hard")
    env.reset()
    return env

# ── Init Tests ────────────────────────────────────────────────────────────────

class TestInit:

    def test_beginner_board_size(self, beginner_env):
        assert beginner_env.rows == 9
        assert beginner_env.cols == 9

    def test_medium_board_size(self, medium_env):
        assert medium_env.rows == 14
        assert medium_env.cols == 14

    def test_hard_board_size(self, hard_env):
        assert hard_env.rows == 24
        assert hard_env.cols == 24

    def test_beginner_mine_count(self, beginner_env):
        assert beginner_env.mine_grid.sum() == 10

    def test_medium_mine_count(self, medium_env):
        assert medium_env.mine_grid.sum() == 25

    def test_hard_mine_count(self, hard_env):
        assert hard_env.mine_grid.sum() == 50

    def test_board_starts_hidden(self, beginner_env):
        assert np.all(beginner_env.visible == HIDDEN)

    def test_no_flags_on_start(self, beginner_env):
        assert beginner_env.correct_flags == 0

    def test_safe_total_correct(self, beginner_env):
        expected = 9 * 9 - 10
        assert beginner_env.safe_total == expected

    def test_action_space_size(self, beginner_env):
        n = 9 * 9
        assert beginner_env.action_space.n == n * 2

# ── Reset Tests ───────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_board(self, beginner_env):
        # Make some moves
        beginner_env.step(0)
        beginner_env.step(1)
        # Reset
        beginner_env.reset()
        assert np.all(beginner_env.visible == HIDDEN)

    def test_reset_clears_flags(self, beginner_env):
        n = beginner_env.rows * beginner_env.cols
        beginner_env.step(n)   # flag cell 0
        beginner_env.reset()
        assert beginner_env.correct_flags == 0

    def test_reset_not_done(self, beginner_env):
        beginner_env.reset()
        assert beginner_env.done == False

    def test_reset_preserves_difficulty(self, beginner_env):
        beginner_env.reset()
        assert beginner_env.difficulty == "Beginner"

    def test_reset_randomizes_mines(self):
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()
        mines1 = env.mine_grid.copy()
        env.reset()
        mines2 = env.mine_grid.copy()
        # Very unlikely to be identical
        assert not np.array_equal(mines1, mines2)

# ── Reveal Tests ──────────────────────────────────────────────────────────────

class TestReveal:

    def test_reveal_mine_ends_game(self):
        """Force a mine hit by revealing all mines."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        # Find a mine position
        mine_positions = np.argwhere(env.mine_grid)
        row, col = mine_positions[0]
        action   = row * env.cols + col

        obs, reward, done, _, _ = env.step(action)
        assert done == True
        assert env.won == False
        assert reward == pytest.approx(-10.0 - 0.1, abs=0.01)

    def test_reveal_safe_cell_positive_reward(self):
        """Reveal a safe cell and check reward."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        # Find a safe cell
        safe_positions = np.argwhere(~env.mine_grid)
        row, col = safe_positions[0]
        action   = row * env.cols + col

        obs, reward, done, _, _ = env.step(action)
        assert done == False or env.won == True
        assert reward > 0 or env.won

    def test_reveal_already_revealed_penalizes(self):
        """Revealing same cell twice should penalize."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        safe_positions = np.argwhere(~env.mine_grid)
        row, col = safe_positions[0]
        action   = row * env.cols + col

        env.step(action)
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_observation_shape(self, beginner_env):
        obs, _ = beginner_env.reset()
        assert obs.shape == (9, 9)

    def test_observation_values_valid(self, beginner_env):
        obs, _ = beginner_env.reset()
        assert np.all(obs >= -3)
        assert np.all(obs <= 8)

# ── Flag Tests ────────────────────────────────────────────────────────────────

class TestFlag:

    def test_correct_flag_rewards(self):
        """Flagging a mine gives positive reward."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        mine_positions = np.argwhere(env.mine_grid)
        row, col = mine_positions[0]
        n        = env.rows * env.cols
        action   = n + row * env.cols + col

        _, reward, _, _, _ = env.step(action)
        assert reward > 0
        assert env.correct_flags == 1

    def test_wrong_flag_penalizes(self):
        """Flagging a safe cell gives negative reward."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        safe_positions = np.argwhere(~env.mine_grid)
        row, col = safe_positions[0]
        n        = env.rows * env.cols
        action   = n + row * env.cols + col

        _, reward, _, _, _ = env.step(action)
        assert reward < 0
        assert env.correct_flags == 0

    def test_unflag_correct_penalizes(self):
        """Removing a correct flag penalizes."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        mine_positions = np.argwhere(env.mine_grid)
        row, col = mine_positions[0]
        n        = env.rows * env.cols
        action   = n + row * env.cols + col

        _, r1, _, _, _ = env.step(action)   # flag it
        assert env.correct_flags == 1       # confirmed flagged
        assert r1 > 0                       # correct flag rewarded

        _, r2, _, _, _ = env.step(action)   # unflag it
        assert r2 < 0                       # penalized for removing

    def test_flag_revealed_cell_penalizes(self):
        """Can't flag an already revealed cell."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        safe_positions = np.argwhere(~env.mine_grid)
        row, col = safe_positions[0]
        n        = env.rows * env.cols

        env.step(row * env.cols + col)           # reveal
        _, reward, _, _, _ = env.step(n + row * env.cols + col)  # flag
        assert reward < 0

# ── Win Condition Tests ───────────────────────────────────────────────────────

class TestWin:

    def test_win_when_all_safe_revealed(self):
        """Reveal all safe cells to win."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        safe_positions = np.argwhere(~env.mine_grid)
        done = False
        won  = False

        for row, col in safe_positions:
            if done:
                break
            action           = row * env.cols + col
            _, _, done, _, _ = env.step(action)
            won              = env.won

        assert won == True

    def test_win_reward_is_positive(self):
        """Win should give large positive reward."""
        env = MinesweeperEnv(render_mode=None, difficulty="Beginner")
        env.reset()

        safe_positions = np.argwhere(~env.mine_grid)
        last_reward    = 0

        for row, col in safe_positions:
            if env.done:
                break
            action                    = row * env.cols + col
            _, last_reward, _, _, _   = env.step(action)

        assert last_reward == pytest.approx(20.0, abs=0.01)

# ── Action Space Tests ────────────────────────────────────────────────────────

class TestActionSpace:

    def test_action_space_samples_valid(self, beginner_env):
        for _ in range(100):
            action = beginner_env.action_space.sample()
            assert 0 <= action < beginner_env.action_space.n

    def test_all_difficulties_have_correct_action_space(self):
        for diff, expected_n in [
            ("Beginner", 9 * 9 * 2),
            ("Medium",   14 * 14 * 2),
            ("Hard",     24 * 24 * 2),
        ]:
            env = MinesweeperEnv(render_mode=None, difficulty=diff)
            env.reset()
            assert env.action_space.n == expected_n
