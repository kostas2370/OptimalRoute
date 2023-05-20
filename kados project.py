import xml.etree.ElementTree as ET
import pprint
import math
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)


def read_data(file_path: str) -> dict:
    dedomena = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    placemarks = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")
    for placemark in placemarks:
        name = placemark.find(".//{http://www.opengis.net/kml/2.2}name").text
        coordinates = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates").text
        coordinate = [float(i) for i in coordinates.split(",")][0:2]
        dedomena[name] = coordinate

    return dedomena


def distance(cords1: list, cords2: list):
   
    d = (math.sqrt((cords2[0] - cords1[0])**2 + (cords2[1] - cords1[1])**2))
    return round(d, 7)


def plot_map(points: dict):
    x = [points[point][0] for point in points]
    y = [points[point][1] for point in points]
    plt.scatter(x, y)
    plt.show()


def make_distance_array(data: list):
    num_points = len(data)
    distance_matrix = np.zeros((num_points, num_points))
    
    for i, point1 in enumerate(data):
        
        for j, point2 in enumerate(data):
            if i == j:
                continue

            distance_matrix[i][j] = distance(data[point1], data[point2])

    np.fill_diagonal(distance_matrix, 0)    
    return distance_matrix


def ac(data: list,
       n_ants: int = 30,
       n_iterations = 100,
       decay: float = 0.5,
       alpha: int= 1,
       beta: int = 2,
       ):

    point_names = [i for i in data]
    print(data)

    distances=make_distance_array(data)

    pheromone = np.ones((len(data), len(data)))
         
    best_path = None
    best_distance = np.inf

    
    for iteration in range(n_iterations):
        # Arxikopoioume ta monopatia twn mirmigkiwn 
        ant_paths = np.zeros((n_ants, len(distances)), dtype=int)
        ant_distances = np.zeros(n_ants)

        # Metakinisi mirmigiou analagoa tin lista twn pheremones kai analoga me tin apostash
        for ant in range(n_ants):
            current_node = np.random.randint(len(distances)) # Vazoume to mirmigki se mia tyxaia thesh
            visited = [current_node]
            #gia ola ta points
            for i in range(len(distances) - 1):
                unvisited = list(set(range(len(distances))) - set(visited)) #pairnoume tin lista twn mh visited shmeiwn
                pheromone_values = np.power(pheromone[current_node, unvisited], alpha)#pairnoume tis times twn pheromones gia ta unvisited node tou twrinou node
                distance_values = np.power(1.0 / distances[current_node, unvisited], beta)#pairnoume tis apostaseis twn apostaseis gia ta unvisited node tou twrinou node
                probabilities = pheromone_values * distance_values / np.sum(pheromone_values * distance_values)*1#pairnoume tis pithanotites gia to epomeno node
                next_node = np.random.choice(unvisited, p=probabilities)
                visited.append(next_node)
                current_node = next_node

            ant_paths[ant] = visited
            ant_distances[ant] += distances[visited[-1], visited[0]]
            for i in range(len(visited) - 1):
                ant_distances[ant] += distances[visited[i], visited[i+1]]

        delta_pheromone = np.zeros(pheromone.shape)
        for ant in range(n_ants):
            for i in range(len(distances) - 1):
                delta_pheromone[ant_paths[ant, i], ant_paths[ant, i+1]] += 1.0 / ant_distances[ant]
            delta_pheromone[ant_paths[ant, -1], ant_paths[ant, 0]] += 1.0 / ant_distances[ant]

        pheromone = (1.0 - decay) * pheromone + delta_pheromone

        if ant_distances.min() < best_distance:
            best_path = ant_paths[ant_distances.argmin()].copy()
            best_distance = ant_distances.min()

        print('iteration {} : {}'.format(iteration,best_distance))

    # Return the best path and distance
    best_path=[point_names[i] for i in best_path]
    return(best_path,best_distance)


data = read_data("panagia.kml")

print(ac(data, n_ants=50, n_iterations = 50))



