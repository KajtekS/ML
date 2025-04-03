import matplotlib.pyplot as plt
from math import sqrt

def rysowanie(points, new_point):
    for point in points:
        if point[2] == "r":
            plt.scatter(point[0], point[1], color='red')
        else:
            plt.scatter(point[0], point[1], color='blue')

        if new_point:
            if new_point[2] == "r":
                plt.scatter(new_point[0], new_point[1], color='red')
            else:
                plt.scatter(new_point[0], new_point[1], color='blue')
    plt.show()

def distance(points, point):
    distances = [0]*len(points)
    i=0

    for x, y, z in points:
        distances[i] = abs(point[0]-x) + abs(point[1]-y)
        i+=1

    for i in range(len(points)):
        points[i].append(distances[i])

points =[
    [40, 20, "r"],
    [50, 50, "b"],
    [60, 90, "b"],
    [10, 25, "r"],
    [70, 70, "b"],
    [60, 10, "r"],
    [25,80, "b"]
]

rysowanie(points, None)

new_point = [35, 50, ""]

distance(points, new_point), points

points = sorted(points, key=lambda point: point[3])

k = 5

il_czerwonych, il_niebieskich = 0, 0

for i in range(k):
    if points[i][2] == "r":
        il_czerwonych += 1
    else:
        il_niebieskich += 1

if il_czerwonych > il_niebieskich:
    new_point = [35, 50, "r"]
    rysowanie(points, new_point)
else:
    new_point = [35, 50, "b"]
    rysowanie(points, new_point)
