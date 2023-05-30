import math
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ant import Ant


def main():
    # Parse the KML file and get the dictionary of placemarks
    placemarks = parse_kml_file("Stena_Panagia___Stena_Profiti_Ilia___Nekrotafeia.kml")

    point_names = list(placemarks.keys())
    points = np.array(list(placemarks.values()))

    best_path_order, monitor_costs = aco(
        points=points,
        alpha=1,
        beta=1,
        evapo_coef=0.05,
        colony_size=500,
        num_iter=30
    )
    print(monitor_costs[-1])

    plot_map(points, best_path_order, monitor_costs)

    sorted_point_names = np.array(point_names)[np.argsort(best_path_order)]

    print(sorted_point_names)


def aco(points, alpha, beta, evapo_coef, colony_size, num_iter):
    # compute (once) the distance matrices
    dist_mat = distance_matrix(points)
    inv_dist_mat = inverse_distance_matrix(points)

    n_locations = points.shape[0]  # total number of points
    ants = [Ant(n_locations) for _ in range(colony_size)]  # ant colony

    # determine initial pheromone value
    phero_init = (inv_dist_mat.mean()) ** (beta / alpha)
    g_phero_graph = np.full((n_locations, n_locations), phero_init)  # pheromone matrix (arbitrary initialization)

    # determine scaling coefficient "Q"
    [ant.ant_trip(g_phero_graph, dist_mat, inv_dist_mat, 1) for ant in ants]
    best_ant = np.argmin([ant.tour_cost for ant in ants])  # ant that scored best in this iteration
    q = ants[best_ant].tour_cost * phero_init / (0.1 * colony_size)

    best_path_length = ants[best_ant].tour_cost
    best_path = ants[best_ant].places_visited.copy()

    monitor_costs = []

    for _ in tqdm(np.arange(num_iter)):

        [ant.ant_trip(g_phero_graph, dist_mat, inv_dist_mat, q) for ant in ants]
        g_phero_graph = update_pheromones(g_phero_graph, ants, evapo_coef).copy()

        iteration_winner = np.argmin([ant.tour_cost for ant in ants])  # ant that scored best in this iteration
        best_path_iteration = ants[iteration_winner].places_visited

        # update global best if better
        if best_path_length > ants[iteration_winner].tour_cost:
            best_path = best_path_iteration.copy()
            best_path_length = ants[iteration_winner].tour_cost

        monitor_costs.append(best_path_length)

        [ant.flush() for ant in ants]

    return best_path, monitor_costs


def distance_matrix(points):
    points_size = points.shape[0]
    dist_mat = np.zeros((points_size, points_size))

    for i in range(points_size):
        for j in range(points_size):
            dist_mat[i, j] = distance(points[i], points[j])

    return dist_mat


def inverse_distance_matrix(points):
    points_size = points.shape[0]
    inv_dist_mat = np.zeros((points_size, points_size))

    # first, construct the distance matrix
    dist_mat = distance_matrix(points)

    for i in range(points_size):
        for j in range(points_size):
            if i == j:
                pass
            else:
                inv_dist_mat[i, j] = 1.0 / dist_mat[i, j]

    return inv_dist_mat


def distance(cords1: list, cords2: list):
    d = (math.sqrt((cords2[0] - cords1[0]) ** 2 + (cords2[1] - cords1[1]) ** 2))
    return round(d, 7)


def update_pheromones(g_phero_graph, ants, evapo_coef=0.05):
    dim = g_phero_graph.shape[0]

    for i in range(dim):
        for j in range(dim):
            g_phero_graph[i, j] = (1 - evapo_coef) * g_phero_graph[i, j] + np.sum(
                [ant.phero_graph[i, j] for ant in ants])
            g_phero_graph[i, j] = max(g_phero_graph[i, j], 1e-08)  # avoid division by zero

    return g_phero_graph


def plot_map(points, best_path, monitor_costs):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(points[best_path, 0],
             points[best_path, 1])

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(monitor_costs)), monitor_costs)

    plt.show()


def parse_kml_file(file_path: str) -> dict:
    placemarks = {}
    try:
        # Parse the KML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find all Placemark elements in the KML
        placemark_elements = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")

        # Process each Placemark and extract relevant information
        for placemark_element in placemark_elements:
            # Extract the necessary information from the Placemark
            name = placemark_element.find(".//{http://www.opengis.net/kml/2.2}name").text
            coord_string = placemark_element.find(".//{http://www.opengis.net/kml/2.2}coordinates").text

            # Add the placemark to the dictionary with name as the key and [lat, long] list as the value
            placemarks[name] = [float(i) for i in coord_string.split(",")][:2]

    except (ET.ParseError, FileNotFoundError) as e:
        # Handle potential exceptions that may occur during parsing
        print("Error parsing KML file:", e)

    return placemarks


if __name__ == "__main__":
    main()
