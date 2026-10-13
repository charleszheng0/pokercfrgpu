import torch

CARDS = ["J", "Q", "K"]
ACTIONS = ["check", "bet", "fold", "call", "raise"]

# Store eight explicit actions. A ninth-action decision is reachable, but its
# omitted tail is uniquely determined by the pot/to-call/raise/player features;
# test_nfsp.py checks that no two reachable information sets collide.
MAX_HISTORY = 8

# Private card (3) + public card (4) + round (2)
# + betting info (4) + history (8 * 9)
INPUT_DIM = 85


def encode_state(obs):
    # Convert the observable poker state into a fixed-length tensor.
    features = []

    features += [float(obs["private_card"] == c) for c in CARDS]

    features += [float(obs["public_card"] == c) for c in CARDS]
    features.append(float(obs["public_card"] is None))

    features += [float(obs["round"] == r) for r in [0, 1]]

    features += [
        obs["pot"] / 30.0,
        obs["to_call"] / 12.0,
        obs["raises_remaining"] / 2.0,
        float(obs["current_player"]),
    ]

    for round_id, player, action in obs["history"][:MAX_HISTORY]:
        features += [float(round_id == r) for r in [0, 1]]
        features += [float(player == p) for p in [0, 1]]
        features += [float(action == a) for a in ACTIONS]

    # Pad shorter histories with zeros.
    features += [0.0] * (INPUT_DIM - len(features))

    assert len(features) == INPUT_DIM

    return torch.tensor(features, dtype=torch.float32)
