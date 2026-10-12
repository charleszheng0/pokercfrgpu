
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==========================================
# 1. GENERATE TRAINING DATA
# ==========================================

X = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = 2 * X**2 + 3 * X + 1

# ==========================================
# 2. DEFINE NEURAL NETWORK
# ==========================================

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNet()

# ==========================================
# 3. DEFINE LOSS + OPTIMIZER
# ==========================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ==========================================
# 4. TRAINING LOOP
# ==========================================

for epoch in range(2000):

    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = loss_fn(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# ==========================================
# 5. TEST THE MODEL
# ==========================================

model.eval()

with torch.no_grad():
    test_X = -2 + 4 * torch.rand(100, 1)
    test_y = 2 * test_X**2 + 3 * test_X + 1

    # Model predictions
    test_predictions = model(test_X)

    # Calculate test error
    test_loss = loss_fn(test_predictions, test_y)

    print(f"Test MSE: {test_loss.item():.6f}")

    # Show 5 sample predictions
    for i in range(5):
        print(
            f"x={test_X[i].item():.2f} | "
            f"Predicted={test_predictions[i].item():.3f} | "
            f"Actual={test_y[i].item():.3f}"
        )
# ==========================================
# 6. VISUALIZE
# ==========================================

with torch.no_grad():
    predicted_y = model(X)

plt.plot(X.numpy(), y.numpy(), label="Actual")
plt.plot(X.numpy(), predicted_y.numpy(), label="Predicted")
plt.legend()
plt.show()
