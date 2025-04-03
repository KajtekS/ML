import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Wczytanie danych
data = pd.read_csv('boston.csv')

# Zmiana nazwy kolumny 'MEDV' na 'PRICE'
data = data.rename(columns={'MEDV': 'PRICE'})

# Przygotowanie zmiennych X i Y
X = data.drop('PRICE', axis=1)  # Cecha
Y = data['PRICE']              # Wartość docelowa

# Podział danych na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Trenowanie modelu
clf = LinearRegression()
clf.fit(x_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = clf.predict(x_test)

# Wykres
plt.figure(figsize=(10, 6))

# Porównanie rzeczywistych i przewidywanych wartości
plt.scatter(y_test, y_pred, color='red', label='Przewidywane wartości')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--', label='Idealne dopasowanie')

# Tytuł i etykiety osi
plt.title('Rzeczywiste vs Przewidywane ceny nieruchomości')
plt.xlabel('Rzeczywiste ceny')
plt.ylabel('Przewidywane ceny')
plt.legend()
plt.show()
