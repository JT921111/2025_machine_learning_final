import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: SDE Simulation ---
def simulate_sde(x0, d, T=1.0, dt=0.01):
    x = x0
    for _ in range(int(T / dt)):
        dw = np.sqrt(dt) * np.random.randn()
        x = x + (-0.3 * x - 0.5 * d) * dt + 0.1 * dw
    return x

# Create dataset
N = 1000
x0_data = np.random.uniform(1, 5, N)
d_data = np.random.uniform(0, 2, N)
xT_data = np.array([simulate_sde(x0_data[i], d_data[i]) for i in range(N)])

X = torch.tensor(np.vstack([x0_data, d_data]).T, dtype=torch.float32)
Y = torch.tensor(xT_data, dtype=torch.float32).unsqueeze(1)

# --- Step 2: MLP Model ---
model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- Step 3: Train ---
losses = []
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# --- Step 4: Plot ---
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
