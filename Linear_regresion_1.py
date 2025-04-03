import torch
from torch import nn
import matplotlib.pyplot as plt

# Parametry do generowania danych
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

# Tworzenie danych
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

# Podział danych na zbiór treningowy i testowy (80/20)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]


# Funkcja do wizualizacji
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Rysuje dane treningowe, testowe oraz przewidywania modelu.
    """
    plt.figure(figsize=(10, 7))

    # Dane treningowe (niebieskie)
    plt.scatter(train_data, train_labels, c="b", s=40, label="Training data")

    # Dane testowe (zielone)
    plt.scatter(test_data, test_labels, c="g", s=40, label="Testing data")

    # Przewidywania modelu (czerwone)
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=40, label="Predictions")

    plt.title("Training, Testing Data and Predictions", fontsize=16)
    plt.xlabel("Input (X)", fontsize=14)
    plt.ylabel("Output (Y)", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()


# Wizualizacja początkowych danych
plot_predictions(X_train, y_train, X_test, y_test)


# Definicja modelu regresji liniowej
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# Inicjalizacja modelu
torch.manual_seed(42)
model_0 = LinearRegression()
print("Początkowe parametry:", list(model_0.parameters()))

# Funkcja straty i optymalizator
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

# Trenowanie modelu
epochs = 10
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()  # Tryb treningu
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Ewaluacja modelu
    model_0.eval()
    with torch.inference_mode():
        test_preds = model_0(X_test)
        test_loss = loss_fn(test_preds, y_test.type(torch.float))

        # Logowanie co 10 epok
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss:.5f} | MAE Test Loss: {test_loss:.5f}")

# Wyświetlenie końcowych parametrów
print("Finalne parametry modelu:", model_0.state_dict())
print("Prawdziwy weight:", weight, "| Prawdziwy bias:", bias)

# Wizualizacja strat
plt.figure(figsize=(10, 7))
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Przewidywania końcowe po treningu
with torch.inference_mode():
    final_preds = model_0(X_test)
plot_predictions(predictions=final_preds)
