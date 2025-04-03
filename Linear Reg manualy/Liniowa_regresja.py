import pandas as pd
import matplotlib.pyplot as plt

def paint(data):
    plt.scatter(data.x,data.y)
    plt.show()

def loss_func(m, b, data):
    E = 0

    for i in range(len(data)):
        E += (data.iloc[i].y - (m * data.iloc[i].x+b))**2
    E /= float(len(data))

    return E

def gradients(m, b, data, L):
    m_gradient = 0
    b_gradient = 0
    n = len(data)

    for i in range(len(data)):
        x = data.iloc[i].x
        y = data.iloc[i].y

        m_gradient += -(2/n)*(y - (m * x + b))*x
        b_gradient += -(2/n)*(y - (m * x + b))

    m = m - L*m_gradient
    b = b - L*b_gradient

    return m, b

data = pd.read_csv("points.csv")

paint(data)
loss_func(0, 0, data)

m=0
b=0
L=0.00026
epochs = 20000

for i in range(epochs):
    m, b = gradients(m, b, data, L)

    if i % 1000 == 0:  # Wydrukuj status co 100 kroków
        current_loss = loss_func(m, b, data)
        print(f"Epoch {i}, Loss: {current_loss}")

print(m, b)

plt.scatter(data.x, data.y, color='black')
plt.plot(list(range(20, 80)), [m*x+b for x in range(20, 80)], color='red')
plt.show()