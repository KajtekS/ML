import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Wczytaj dane
dataset = pd.read_csv('austin_weather.csv')

# Usuń niepotrzebne kolumny
dataset = dataset.drop(['Events', 'Date', 'SeaLevelPressureHighInches',
                        'SeaLevelPressureLowInches'], axis=1)

# Zamień wartości 'T' i '-' na 0.0
dataset = dataset.replace('T', 0.0)
dataset = dataset.replace('-', 0.0)

# Upewnij się, że wszystkie dane są typu float
dataset = dataset.astype(float)

# Przygotowanie zmiennych X i Y
X = dataset.drop(['PrecipitationSumInches'], axis=1)
Y = dataset['PrecipitationSumInches']

# Przekształcenie Y na macierz kolumnową
Y = Y.values.reshape(-1, 1)

# Utworzenie i dopasowanie modelu
clf = LinearRegression()
clf.fit(X, Y)

# Indeks dnia do zaznaczenia
day_index = 760

# Lista dni (od 0 do liczby próbek - 1)
days = [i for i in range(Y.size)]

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.scatter(days, Y, color='g', label='Wartości opadów')
if day_index < len(days):
    plt.scatter(days[day_index], Y[day_index], color='r', label=f'Dzień {day_index}')
else:
    print(f"Warning: day_index ({day_index}) exceeds the data range.")
plt.title('Rozkład opadów w czasie')
plt.xlabel('Indeks dnia')
plt.ylabel('Suma opadów (cale)')
plt.legend()
plt.show()
