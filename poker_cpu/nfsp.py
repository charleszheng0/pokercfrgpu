"""CPU NFSP self-play on 20-chip restricted heads-up Hold'em.

This is a separate agent from the Leduc NFSP experiment. No Leduc network or
hidden opponent cards are used. A fixed 11-slot action vocabulary represents
fold/check/call and up to four bet or raise sizes, with legal masks and actual
wager amounts included in the observation features.
"""

import random
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from .cards import DECK
from .engine import HoldemGame

STACK = 20
MAX_HISTORY = 32
KINDS = ("fold", "check", "call", "bet", "raise")
ACTION_COUNT = 11
INPUT_DIM = 525


def indexed_actions(legal):
    result = {}
    for kind, index in (("fold", 0), ("check", 1), ("call", 2)):
        action = next((a for a in legal if a.kind == kind), None)
        if action is not None:
            result[index] = action
    for kind, first, last in (("bet", 3, 6), ("raise", 7, 10)):
        choices = [a for a in legal if a.kind == kind]
        if len(choices) == 1:
            result[last] = choices[0]
        elif choices:
            result[first] = choices[0]
            result[last] = choices[-1]
            for offset, action in enumerate(choices[1:-1], start=1):
                result[first + offset] = action
    return result


def encode(obs):
    own = set(obs["hole"])
    board = set(obs["board"])
    features = [float(card in own) for card in DECK]
    features.extend(float(card in board) for card in DECK)
    features.extend(float(obs["street"] == street) for street in range(4))
    features.extend(float(obs["player"] == player) for player in (0, 1))
    features.append(float(obs["button"] == obs["player"]))
    features.extend(stack / STACK for stack in obs["stacks"])
    features.append(obs["pot"] / (2 * STACK))
    features.append(obs["to_call"] / STACK)
    features.extend(paid / STACK for paid in obs["street_paid"])
    features.extend(paid / STACK for paid in obs["paid"])
    if len(obs["history"]) > MAX_HISTORY:
        raise ValueError("History exceeds complete fixed-size encoding")
    for street, player, kind, amount in obs["history"]:
        features.extend(float(street == i) for i in range(4))
        features.extend(float(player == i) for i in (0, 1))
        features.extend(float(kind == choice) for choice in KINDS)
        features.append(amount / STACK)
    features.extend([0.0] * ((MAX_HISTORY - len(obs["history"])) * 12))
    indexed = indexed_actions(obs["legal_actions"])
    features.extend(
        indexed[i].amount / STACK if i in indexed else 0.0 for i in range(ACTION_COUNT)
    )
    features.extend(float(i in indexed) for i in range(ACTION_COUNT))
    if len(features) != INPUT_DIM:
        raise AssertionError(f"Expected {INPUT_DIM} features, got {len(features)}")
    return torch.tensor(features, dtype=torch.float32)


def network():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, ACTION_COUNT),
    )


class Reservoir:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.data = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            index = random.randrange(self.seen)
            if index < self.capacity:
                self.data[index] = item

    def sample(self, size):
        return random.sample(self.data, size)


class HoldemNFSPAgent:
    def __init__(self, eta=0.3, epsilon=0.1, lr=0.001):
        self.eta, self.epsilon = eta, epsilon
        self.q = network()
        self.target = network()
        self.target.load_state_dict(self.q.state_dict())
        self.policy = network()
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.rl = deque(maxlen=100000)
        self.sl = Reservoir()
        self.q_updates = 0
        self.mode = "average"

    def start_hand(self):
        self.mode = "best_response" if random.random() < self.eta else "average"

    def act(self, obs, record=True):
        state = encode(obs)
        indexed = indexed_actions(obs["legal_actions"])
        available = list(indexed)
        with torch.no_grad():
            if self.mode == "best_response":
                if random.random() < self.epsilon:
                    index = random.choice(available)
                else:
                    scores = self.q(state)
                    index = max(available, key=lambda i: scores[i].item())
                if record:
                    mask = torch.tensor([i in indexed for i in range(ACTION_COUNT)])
                    self.sl.add((state.clone(), index, mask))
            else:
                logits = self.policy(state)
                mask = torch.tensor([i in indexed for i in range(ACTION_COUNT)])
                probs = torch.softmax(logits.masked_fill(~mask, -1e9), dim=0)
                index = torch.multinomial(probs, 1).item()
        return index, indexed[index]

    def update(self, batch_size=64):
        q_loss = sl_loss = None
        if len(self.rl) >= batch_size:
            batch = random.sample(self.rl, batch_size)
            states = torch.stack([row[0] for row in batch])
            actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
            rewards = torch.tensor(
                [row[2] / STACK for row in batch], dtype=torch.float32
            )
            next_states = torch.stack([row[3] for row in batch])
            done = torch.tensor([row[4] for row in batch], dtype=torch.bool)
            next_legal = torch.stack([row[5] for row in batch])
            q_values = self.q(states).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                future = (
                    self.target(next_states)
                    .masked_fill(~next_legal, -1e9)
                    .max(1)
                    .values
                )
                future = torch.where(done, torch.zeros_like(future), future)
                target = rewards + future
            loss = F.smooth_l1_loss(q_values, target)
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()
            q_loss = loss.item()
            self.q_updates += 1
            if self.q_updates % 200 == 0:
                self.target.load_state_dict(self.q.state_dict())
        if len(self.sl.data) >= batch_size:
            batch = self.sl.sample(batch_size)
            states = torch.stack([row[0] for row in batch])
            actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
            masks = torch.stack([row[2] for row in batch])
            logits = self.policy(states).masked_fill(~masks, -1e9)
            loss = F.cross_entropy(logits, actions)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            sl_loss = loss.item()
        return q_loss, sl_loss


def play_self_play_hand(agents, seed, button):
    game = HoldemGame(stacks=(STACK, STACK), button=button, seed=seed)
    for agent in agents:
        agent.start_hand()
    pending = [None, None]
    while not game.done:
        player = game.current_player
        agent = agents[player]
        obs = game.observe(player)
        state = encode(obs)
        if pending[player] is not None:
            old_state, old_index = pending[player]
            legal = indexed_actions(obs["legal_actions"])
            mask = torch.tensor([i in legal for i in range(ACTION_COUNT)])
            agent.rl.append((old_state, old_index, 0.0, state.clone(), False, mask))
        index, action = agent.act(obs)
        pending[player] = (state.clone(), index)
        game.step(action)
    rewards = game.rewards()
    for player, previous in enumerate(pending):
        if previous is not None:
            old_state, old_index = previous
            agents[player].rl.append(
                (
                    old_state,
                    old_index,
                    float(rewards[player]),
                    torch.zeros(INPUT_DIM),
                    True,
                    torch.zeros(ACTION_COUNT, dtype=torch.bool),
                )
            )
    return rewards
