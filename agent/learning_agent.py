# learning_agent.py
# The DataForge Learning Agent — uses Deep Q-Learning (DQN) to learn Minesweeper
#
# HOW IT WORKS IN PLAIN ENGLISH:
# The agent plays thousands of games and learns from experience.
# It has a "brain" (neural network) that looks at the board and decides
# which cell to click. At first it's random. Over time it learns which
# moves lead to good outcomes (revealing safe cells) and which lead to
# bad outcomes (hitting mines).
#
# This is called Deep Q-Learning (DQN) — the same algorithm DeepMind
# used to beat Atari games in 2013. You're building that from scratch.
#
# Run this with:  python agent/learning_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
from collections import deque
from games.minesweeper.minesweeper_env import MinesweeperEnv
from agent.dashboard import Dashboard

# ── Device Setup ─────────────────────────────────────────────────────────────
# This automatically uses the GPU (Matt's GTX 1050 Ti) if available,
# otherwise falls back to CPU. You should see "Using device: cuda" in terminal.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
# These are the "settings" that control how the agent learns.
# Think of them like process parameters in manufacturing —
# tuning these changes the quality and speed of learning.

EPISODES        = 2000    # total games to play
BATCH_SIZE      = 64      # how many memories to learn from at once
GAMMA           = 0.95    # how much to value future rewards vs immediate ones
                          # (0 = only care about now, 1 = care about distant future)
EPSILON_START   = 1.0     # starting exploration rate (1.0 = 100% random at first)
EPSILON_END     = 0.05    # minimum exploration rate (always explore a little)
EPSILON_DECAY   = 0.995   # how fast to reduce randomness each episode
LEARNING_RATE   = 0.001   # how fast the neural network updates its weights
MEMORY_SIZE     = 10000   # how many past experiences to remember
TARGET_UPDATE   = 10      # how often to update the target network (in episodes)
STEP_DELAY      = 0.05    # seconds between moves (reduce to train faster)


# ── The Brain: Neural Network ─────────────────────────────────────────────────
# This is the agent's decision-making engine.
# It takes the board state as input and outputs a score for every possible
# cell click. The agent picks the cell with the highest score.
#
# Think of it like a function: board state → action scores
# It learns by adjusting its internal weights every time it makes a mistake.

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()

        # Three layers of neurons — input, hidden, output
        # Each layer learns increasingly abstract patterns:
        # Layer 1: raw board values
        # Layer 2: patterns between cells (clusters, edges)
        # Layer 3: strategic understanding (safe zones, mine probability)
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),   # input layer → 256 neurons
            nn.ReLU(),                     # activation: "fire if positive"
            nn.Linear(256, 256),          # hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),          # another hidden layer
            nn.ReLU(),
            nn.Linear(128, output_size)   # output: one score per cell
        )

    def forward(self, x):
        return self.network(x)


# ── Memory: Experience Replay Buffer ─────────────────────────────────────────
# The agent doesn't learn from each move immediately.
# Instead it stores experiences (like taking notes) and later
# randomly samples from them to learn. This breaks correlations
# between consecutive moves and makes learning more stable.
#
# Each memory = (board_before, action_taken, reward, board_after, done?)
# LSS analogy: this is your defect log — you review past events
# to find patterns, not just react to the last thing that happened.

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ── The Agent ─────────────────────────────────────────────────────────────────
# This ties everything together — the brain, the memory, and the
# decision-making logic.

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON_START    # current exploration rate

        # Two identical networks — this is a DQN trick for stability.
        # "policy_net" learns constantly.
        # "target_net" is a frozen copy updated every TARGET_UPDATE episodes.
        # Learning against a moving target is unstable — like trying to
        # hit a bullseye that keeps moving. The target net holds still.
        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=LEARNING_RATE)
        self.memory    = ReplayBuffer(MEMORY_SIZE)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, state, env):
        """
        Epsilon-greedy action selection.
        With probability epsilon → pick a random cell (explore)
        Otherwise → ask the neural network (exploit what we know)

        Early in training epsilon is high so the agent explores a lot.
        Over time epsilon decays and the agent trusts its own judgment more.
        This is the explore vs exploit tradeoff — a core concept in RL.
        """
        if random.random() < self.epsilon:
            return env.action_space.sample()   # random exploration

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            q_values     = self.policy_net(state_tensor)
            return q_values.argmax().item()    # best known action

    def learn(self):
        """
        Sample a batch of memories and update the neural network.
        This is where actual learning happens.

        The agent asks: "given what I knew then, was my action good?"
        It compares what it predicted would happen vs what actually happened,
        then nudges the network weights to be more accurate next time.
        """
        if len(self.memory) < BATCH_SIZE:
            return   # not enough memories yet — keep playing

        batch      = self.memory.sample(BATCH_SIZE)
        states     = torch.FloatTensor(
                         np.array([b[0].flatten() for b in batch])).to(device)
        actions    = torch.LongTensor([b[1] for b in batch]).to(device)
        rewards    = torch.FloatTensor([b[2] for b in batch]).to(device)
        next_states= torch.FloatTensor(
                         np.array([b[3].flatten() for b in batch])).to(device)
        dones      = torch.FloatTensor([b[4] for b in batch]).to(device)

        # What did our network predict for these actions?
        current_q  = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # What does the target network say the best future reward would be?
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        # How wrong were we? Compute loss and backpropagate
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """Decay exploration rate — get less random over time."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="agent/saved_model.pth"):
        """Save the trained model so we don't lose progress."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="agent/saved_model.pth"):
        """Load a previously trained model."""
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {path}")
        else:
            print("No saved model found — starting fresh.")


# ── Main Training Loop ────────────────────────────────────────────────────────
# This is the heartbeat of the whole system.
# Episode after episode, the agent plays, remembers, and learns.
# Watch the dashboard — the win rate line should start climbing
# after a few hundred episodes as the agent figures things out.

def run():
    env        = MinesweeperEnv(render_mode="human")
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size= env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load()   # load saved progress if it exists

    dash  = Dashboard(agent_name="DQN Learning Agent", color="#E15759")

    print(f"\nStarting training for {EPISODES} episodes...")
    print(f"Watch the dashboard — win rate should climb over time!\n")

    for episode in range(1, EPISODES + 1):
        obs, _       = env.reset()
        state        = obs
        total_reward = 0
        done         = False
        step         = 0

        while not done:
            # Handle window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save()
                    env.close()
                    dash.close()
                    return

            # Agent picks an action
            action = agent.select_action(state, env)

            # Environment responds
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs

            # Store this experience in memory
            agent.memory.push(state, action, reward, next_state, done)

            # Learn from a random batch of past experiences
            agent.learn()

            state         = next_state
            total_reward += reward
            step         += 1

            import time
            time.sleep(STEP_DELAY)

        # Episode complete
        agent.update_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 50 == 0:
            agent.save()   # save progress every 50 episodes

        result = "WIN 🎉" if env.won else "LOSS 💥"
        print(f"Episode {episode:4d} | {result} | "
              f"Steps: {step:3d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Epsilon: {agent.epsilon:.3f}")

        dash.update(
            episode = episode,
            reward  = total_reward,
            won     = env.won,
            steps   = step
        )

    agent.save()
    env.close()
    dash.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    run()
