# learning_agent.py
# DQN Learning Agent — Version 2.0
# Now with:
#   - Difficulty selection screen
#   - Expanded action space (reveal + flag)
#   - Updated reward tracking (flags, board cleared %)
#   - Shared dashboard for comparison with random agent
#   - Model saves per difficulty level
#
# Run this with:  python agent/learning_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
from collections import deque
from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.dashboard import Dashboard

# ── Device Setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# These control how the agent learns.
# Think of them like process parameters — tuning changes quality and speed.

EPISODES        = 3000    # total games to play
BATCH_SIZE      = 64      # memories to learn from at once
GAMMA           = 0.95    # future reward discount
                          # (0 = only now, 1 = care about distant future)
EPSILON_START   = 1.0     # start fully random
EPSILON_END     = 0.05    # always keep a little randomness
EPSILON_DECAY   = 0.997   # slower decay — more exploration time
LEARNING_RATE   = 0.0005  # how fast the network updates
MEMORY_SIZE     = 20000   # how many experiences to remember
TARGET_UPDATE   = 20      # episodes between target network updates
SAVE_EVERY      = 100     # episodes between model saves
STEP_DELAY      = 0.03    # seconds between moves (lower = faster training)


# ── Neural Network ────────────────────────────────────────────────────────────
# The agent's brain.
# Input  = flattened board state
# Output = score for every possible action (reveal or flag each cell)
#
# Deeper network than v1 to handle the larger action space and
# more complex reward structure including flagging.

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ── Experience Replay Buffer ──────────────────────────────────────────────────
# Stores past experiences so the agent can learn from them later.
# Random sampling breaks correlations between consecutive moves
# and makes learning more stable.
#
# Each memory = (state, action, reward, next_state, done)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
# Ties together the brain, memory, and decision logic.

class DQNAgent:
    def __init__(self, state_size, action_size, difficulty="Beginner"):
        self.state_size  = state_size
        self.action_size = action_size
        self.difficulty  = difficulty
        self.epsilon     = EPSILON_START

        # Two networks for stability:
        # policy_net  = learns constantly
        # target_net  = frozen copy, updated every TARGET_UPDATE episodes
        # Learning against a moving target is unstable — target_net holds still
        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory    = ReplayBuffer(MEMORY_SIZE)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, state, env):
        """
        Epsilon-greedy action selection.
        High epsilon = random exploration (early training)
        Low epsilon  = trust the network (later training)

        Also avoids obviously bad actions:
        — never try to reveal an already revealed cell
        — never try to flag an already revealed cell
        This is called action masking and speeds up learning significantly.
        """
        # Build mask of valid actions
        n           = env.rows * env.cols
        flat_board  = state.flatten()
        valid       = []

        for i in range(n):
            if flat_board[i] == -1:      # hidden — can reveal or flag
                valid.append(i)          # reveal
                valid.append(i + n)      # flag
        if not valid:
            valid = list(range(env.action_space.n))

        if random.random() < self.epsilon:
            return random.choice(valid)   # explore

        with torch.no_grad():
            state_t  = torch.FloatTensor(
                flat_board).unsqueeze(0).to(device)
            q_values = self.policy_net(state_t).squeeze()

            # Mask invalid actions with very low value
            mask     = torch.full(
                (env.action_space.n,), float("-inf")).to(device)
            for v in valid:
                mask[v] = q_values[v]

            return mask.argmax().item()   # exploit

    def learn(self):
        """
        Sample a batch of memories and update the network.
        Compares predicted Q-values to target Q-values,
        computes loss, and backpropagates to improve predictions.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        batch       = self.memory.sample(BATCH_SIZE)
        states      = torch.FloatTensor(
            np.array([b[0].flatten() for b in batch])).to(device)
        actions     = torch.LongTensor(
            [b[1] for b in batch]).to(device)
        rewards     = torch.FloatTensor(
            [b[2] for b in batch]).to(device)
        next_states = torch.FloatTensor(
            np.array([b[3].flatten() for b in batch])).to(device)
        dones       = torch.FloatTensor(
            [b[4] for b in batch]).to(device)

        current_q   = self.policy_net(states).gather(
            1, actions.unsqueeze(1))

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        # Like a safety valve on a pressure system
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_epsilon(self):
        """Decay exploration rate — trust the network more over time."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        """Sync target network with policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        """Save model — separate file per difficulty so progress isn't mixed."""
        path = f"agent/model_{self.difficulty.lower()}.pth"
        torch.save({
            "model_state":     self.policy_net.state_dict(),
            "epsilon":         self.epsilon,
            "difficulty":      self.difficulty,
        }, path)
        print(f"  💾 Model saved → {path}")

    def load(self):
        """Load saved model for this difficulty if it exists."""
        path = f"agent/model_{self.difficulty.lower()}.pth"
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.policy_net.load_state_dict(checkpoint["model_state"])
            self.target_net.load_state_dict(checkpoint["model_state"])
            self.epsilon = checkpoint.get("epsilon", EPSILON_START)
            print(f"  ✅ Model loaded ← {path}")
            print(f"  Resuming at epsilon: {self.epsilon:.3f}")
        else:
            print(f"  🆕 No saved model found — starting fresh.")


# ── Main Training Loop ────────────────────────────────────────────────────────

def run(game_name="Minesweeper"):
    env    = MinesweeperEnv(render_mode="human")
    obs, _ = env.reset()   # triggers difficulty selection screen

    state_size  = env.rows * env.cols
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, difficulty=env.difficulty)
    agent.load()

    dash = Dashboard(agent_name="DQN Learning Agent")

    print(f"\nDQN Agent starting on {env.difficulty}...")
    print(f"Board: {env.rows}x{env.cols} | Mines: {env.num_mines}")
    print(f"State size: {state_size} | Action size: {action_size}")
    print(f"Starting epsilon: {agent.epsilon:.3f}")
    print(f"Training for {EPISODES} episodes...\n")

    for episode in range(1, EPISODES + 1):
        if episode > 1:
            obs, _ = env.reset()

        state        = obs
        total_reward = 0
        done         = False
        step         = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save()
                    env.close()
                    dash.close()
                    return

            action = agent.select_action(state, env)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state        = next_state
            total_reward += reward
            step         += 1

            time.sleep(STEP_DELAY)

        agent.update_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % SAVE_EVERY == 0:
            agent.save()

        result = "WIN 🎉" if env.won else "LOSS 💥"
        print(f"Episode {episode:4d} | {result} | "
              f"Steps: {step:3d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Flags: {env.correct_flags}/{env.num_mines} | "
              f"Cleared: {env.safe_revealed}/{env.safe_total} | "
              f"ε: {agent.epsilon:.3f}")

        dash.update(
            episode       = episode,
            reward        = total_reward,
            won           = env.won,
            steps         = step,
            correct_flags = env.correct_flags,
            pct_cleared   = env.safe_revealed / env.safe_total
                            if env.safe_total > 0 else 0.0
        )

    agent.save()
    env.close()
    dash.close()
    print("\nTraining complete!")


def run_headless(game_name="Minesweeper", dash=None,
                 stop_flag=None, watch_request=None):
    """
    Runs learning agent — renders only when watch_request matches.
    Used by launcher threading architecture.
    """
    import time
    from games.registry import GAME_REGISTRY

    cfg      = GAME_REGISTRY[game_name]
    EnvClass = cfg["env_class"]

    if stop_flag is None:
        stop_flag = {"done": False}

    temp_env    = EnvClass(render_mode=None, difficulty="Beginner")
    temp_obs, _ = temp_env.reset()
    state_size  = temp_env.rows * temp_env.cols
    action_size = temp_env.action_space.n
    diff        = temp_env.difficulty
    temp_env.close()

    agent   = DQNAgent(state_size, action_size, difficulty=diff)
    agent.load()

    episode = 0
    TOTAL   = EPISODES

    for _ in range(TOTAL):
        if stop_flag.get("done"):
            break
        if watch_request and watch_request.get("agent") == "MENU":
            break

        watching = (watch_request and
                    watch_request.get("agent") == "DQN Learning Agent")

        env      = EnvClass(
            render_mode="human" if watching else None,
            difficulty=diff
        )
        obs, _   = env.reset()
        state    = obs
        done     = False
        reward   = 0
        steps    = 0

        while not done:
            action = agent.select_action(state, env)
            next_s, r, done, _, _ = env.step(action)
            agent.memory.push(state, action, r, next_s, done)
            agent.learn()
            state   = next_s
            reward += r
            steps  += 1
            if watching:
                time.sleep(STEP_DELAY)

        env.close()
        agent.update_epsilon()
        episode += 1

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        if episode % SAVE_EVERY == 0:
            agent.save()

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

        print(f"[Learning] Ep {episode:4d} | "
              f"{'WIN 🎉' if env.won else 'LOSS 💥'} | "
              f"Steps: {steps:3d} | Reward: {reward:6.1f} | "
              f"ε: {agent.epsilon:.3f}")

    agent.save()

if __name__ == "__main__":
    run()
  
