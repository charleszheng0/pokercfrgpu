"""Original PyTorch regression exercise, kept separate from poker training."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


def main():
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 200).reshape(-1, 1)
    y = 2 * x**2 + 3 * x + 1
    model = NeuralNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_x = -2 + 4 * torch.rand(100, 1)
        test_y = 2 * test_x**2 + 3 * test_x + 1
        test_predictions = model(test_x)
        print(f"Test MSE: {loss_fn(test_predictions, test_y).item():.6f}")
        for i in range(5):
            print(
                f"x={test_x[i].item():.2f} | "
                f"Predicted={test_predictions[i].item():.3f} | "
                f"Actual={test_y[i].item():.3f}"
            )
        predicted_y = model(x)

    plt.plot(x.numpy(), y.numpy(), label="Actual")
    plt.plot(x.numpy(), predicted_y.numpy(), label="Predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
