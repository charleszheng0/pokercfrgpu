import torch
import torch.nn as nn
from .encoder import INPUT_DIM

ACTIONS = ["check", "bet", "fold", "call", "raise"]


class QNetwork(nn.Module):
    def __init__(self, input_dim):
        # Build a neural network predicting Q-values for five poker actions.
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, x):
        # Return predicted Q-values for every action.
        return self.net(x)


if __name__ == "__main__":
    # Test the network with the correct number of input features.
    q_network = QNetwork(input_dim=INPUT_DIM)

    dummy_state = torch.randn(INPUT_DIM)

    with torch.no_grad():
        q_values = q_network(dummy_state)

    for action, value in zip(ACTIONS, q_values):
        print(f"{action}: {value.item():.4f}")
