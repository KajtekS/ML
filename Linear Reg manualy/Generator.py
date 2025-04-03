import csv
import random


def generate_data_on_line_with_noise(file_name, num_points, m, b, noise_level):
    # Otwarcie pliku CSV do zapisu
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Zapisanie nagłówków
        writer.writerow(['x', 'y'])

        # Generowanie i zapisywanie danych wzdłuż linii z szumem
        for _ in range(num_points):
            x = round(random.uniform(0, 100), 2)  # Losowa liczba x w zakresie 0-100
            # Obliczanie y na podstawie linii + dodanie szumu
            y = round(m * x + b + random.uniform(-noise_level, noise_level), 2)
            writer.writerow([x, y])


# Parametry prostej
m = 0.5  # Współczynnik kierunkowy (nachylenie)
b = 10  # Punkt przecięcia z osią y
noise_level = 5  # Poziom szumu (rozproszenie punktów wokół linii)

# Generowanie danych i zapisanie do pliku
generate_data_on_line_with_noise('points.csv', 100, m, b, noise_level)
