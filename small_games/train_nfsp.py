"""Original NFSP experiment on the repository's custom Leduc rules."""

import argparse
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .leduc import LeducPoker
from .dqn import QNetwork, ACTIONS
from .encoder import encode_state, INPUT_DIM


class ReplayBuffer:
    def __init__(self, capacity=10000):
        # Store recent transitions for DQN training.
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    def __init__(self, capacity=10000):
        # Maintain a uniform sample of historical best-response actions.
        self.capacity = capacity
        self.buffer = []
        self.seen = 0

    def push(self, example):
        self.seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(example)
        else:
            index = random.randrange(self.seen)

            if index < self.capacity:
                self.buffer[index] = example

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class AveragePolicyNetwork(nn.Module):
    def __init__(self):
        # Predict action logits used to form the average strategy.
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, state):
        return self.net(state)


class NFSPAgent:
    def __init__(self, eta=0.1, epsilon=0.1, lr=0.001, reward_scale=1.0):
        # Create both learning systems for one poker player.
        self.eta = eta
        self.epsilon = epsilon
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.reward_scale = reward_scale

        self.q_network = QNetwork(INPUT_DIM)
        self.target_network = QNetwork(INPUT_DIM)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.policy_network = AveragePolicyNetwork()

        self.rl_buffer = ReplayBuffer()
        self.sl_buffer = ReservoirBuffer()

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr
        )

        self.mode = "average"
        self.q_updates = 0

    def start_episode(self):
        # Eta selects the behavioral policy for this entire hand.
        self.mode = "best_response" if random.random() < self.eta else "average"

    def choose_action(self, obs, legal_actions):
        # Select a legal action and record best-response decisions.
        state = encode_state(obs)
        legal_indices = [ACTIONS.index(a) for a in legal_actions]

        with torch.no_grad():
            if self.mode == "best_response":
                if random.random() < self.epsilon:
                    action_index = random.choice(legal_indices)
                else:
                    q_values = self.q_network(state)
                    action_index = max(legal_indices, key=lambda i: q_values[i].item())

                legal_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
                legal_mask[legal_indices] = True

                self.sl_buffer.push((state.clone(), action_index, legal_mask))

            else:
                logits = self.policy_network(state)

                mask = torch.full_like(logits, -1e9)
                mask[legal_indices] = 0

                probabilities = F.softmax(logits + mask, dim=-1)
                action_index = torch.multinomial(probabilities, 1).item()

        return ACTIONS[action_index]

    def train_dqn(self, batch_size=64, gamma=1.0):
        # Update the online Q-network using Bellman targets.
        if len(self.rl_buffer) < batch_size:
            return None

        batch = self.rl_buffer.sample(batch_size)

        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor(
            [t[2] / self.reward_scale for t in batch], dtype=torch.float32
        )
        next_states = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.bool)
        next_legal = torch.stack([t[5] for t in batch])

        predicted_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            future_q = self.target_network(next_states)
            future_q = future_q.masked_fill(~next_legal, -1e9)

            next_best = future_q.max(dim=1).values
            next_best = torch.where(dones, torch.zeros_like(next_best), next_best)

            bellman_targets = rewards + gamma * next_best

        loss = F.smooth_l1_loss(predicted_q, bellman_targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.q_updates += 1

        if self.q_updates % 200 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train_average_policy(self, batch_size=64):
        # Imitate historical best-response actions using cross-entropy.
        if len(self.sl_buffer) < batch_size:
            return None

        batch = self.sl_buffer.sample(batch_size)

        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        legal_masks = torch.stack([t[2] for t in batch])

        logits = self.policy_network(states)
        logits = logits.masked_fill(~legal_masks, -1e9)

        loss = F.cross_entropy(logits, actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()


def play_hand(game, agents):
    # Play one hand and construct transitions between each player's turns.
    game.reset()

    for agent in agents:
        agent.start_episode()

    pending = [None, None]

    while not game.done:
        player = game.current_player
        agent = agents[player]

        obs = game.observe(player)
        state = encode_state(obs)
        legal = game.legal_actions()

        # Resolve this player's previous action when they act again.
        if pending[player] is not None:
            old_state, old_action = pending[player]

            legal_mask = torch.tensor([a in legal for a in ACTIONS], dtype=torch.bool)

            agent.rl_buffer.push(
                (old_state, old_action, 0.0, state.clone(), False, legal_mask)
            )

            pending[player] = None

        action = agent.choose_action(obs, legal)

        # Remember the action until this player acts again
        # or the poker hand terminates.
        pending[player] = (state.clone(), ACTIONS.index(action))

        game.step(action)

    rewards = game.rewards()

    # Resolve each player's last decision using the final chip payoff.
    for player, previous in enumerate(pending):
        if previous is None:
            continue

        old_state, old_action = previous

        agents[player].rl_buffer.push(
            (
                old_state,
                old_action,
                float(rewards[player]),
                torch.zeros(INPUT_DIM),
                True,
                torch.zeros(len(ACTIONS), dtype=torch.bool),
            )
        )

    return rewards


def train_nfsp(num_episodes=100000, batch_size=64, output_dir=None):
    # Generate self-play experience and train both networks repeatedly.
    game = LeducPoker()

    agents = [NFSPAgent(), NFSPAgent()]

    reward_totals = [0.0, 0.0]

    for episode in range(1, num_episodes + 1):
        rewards = play_hand(game, agents)

        for player in range(2):
            reward_totals[player] += rewards[player]

        # Train periodically after collecting sufficient experience.
        if episode % 4 == 0:
            for agent in agents:
                agent.train_dqn(batch_size)
                agent.train_average_policy(batch_size)

        if episode % 1000 == 0:
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Average rewards: "
                f"P0={reward_totals[0]/episode:.3f}, "
                f"P1={reward_totals[1]/episode:.3f} | "
                f"RL buffer={len(agents[0].rl_buffer)} | "
                f"SL buffer={len(agents[0].sl_buffer)}"
            )

    # Keep new training outputs separate from pre-existing legacy checkpoints.
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "nfsp"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save both players' learned average strategies.
    for player, agent in enumerate(agents):
        torch.save(
            agent.policy_network.state_dict(), output_dir / f"nfsp_player_{player}.pt"
        )

        torch.save(
            agent.q_network.state_dict(), output_dir / f"nfsp_q_player_{player}.pt"
        )

    print("Training complete. Models saved.")
    return agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_nfsp(args.episodes, args.batch_size, args.output_dir)
