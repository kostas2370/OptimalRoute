import xml.etree.ElementTree as ET
import pprint
import math
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

num_iterations = 100
alpha = 10
beta = 0.005
evaporation_rate = 0.5
pheromone_constant = 100

def read_data(file_path: str) -> dict:
    dedomena = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    placemarks = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")
    for placemark in placemarks:
        name = placemark.find(".//{http://www.opengis.net/kml/2.2}name").text
        coordinates = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates").text
        coordinate = [float(i) for i in coordinates.split(",")]
        dedomena[name]=coordinate

    return dedomena

def distance(cords1: list,cords2: list):
   
    d = math.sqrt((cords2[0] - cords1[0])**2 + (cords2[1] - cords1[1])**2 + (cords2[2] - cords1[2])**2)
    return round(d,3)

def plot_map(points: dict):
    x = [points[point][0] for point in points]
    y = [points[point][1] for point in points]
    z = [points[point][2] for point in points]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def make_distance_array(data):
    num_points = len(data)
    distance_matrix = np.zeros((num_points,num_points))
    
    for i,point1 in enumerate(data):
        
        for j, point2 in enumerate(data):
            if i == j:
                continue

            distance_matrix[i][j] = distance(data[point1], data[point2])
    np.fill_diagonal(distance_matrix, 0)    
    return distance_matrix



data = read_data("vyrwnas.kml")
plot_map(data)
distance_array=make_distance_array(data)
print(data)
pheromone_matrix = np.ones((len(data), len(data)))




#print(pheromone_matrix[0][3] ** alpha) * ((1 / distance_array[0][3]) ** beta)
#print(distance(data["173"],data["174"]))
