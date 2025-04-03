import torch
from torch import nn
import matplotlib.pyplot as plt


def rysowanie(X_train, Y_train, X_test, Y_test, predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(X_train, Y_train, c='blue', marker='o', s=5)
    plt.scatter(X_test, Y_test, c='red', marker='o', s=5)
    if predictions is not None:
        plt.scatter(X_test, predictions, c='green', marker='o', s=5)
    plt.grid(True)
    plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=float, requires_grad=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

weight = 0.9
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight*X + bias

train_len = int(len(X)*0.8)
X_train, Y_train = X[:train_len], Y[:train_len]
X_test, Y_test = X[train_len:], Y[train_len:]

rysowanie(X_train, Y_train, X_test, Y_test, predictions=None)

model = LinearRegression()

loss_fn = nn.L1Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

epochs = 100

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        loss = loss_fn(test_pred, Y_test)

with torch.inference_mode():
    final_preds = model(X_test)
rysowanie(predictions=final_preds)
