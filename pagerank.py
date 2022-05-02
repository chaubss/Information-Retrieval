import numpy as np
import sys, getopt
from tabulate import tabulate

ALPHA = 0.1
BREAK_THRESHOLD = 0.0000000001


def generate_probability_transition_matrix(nodes_count, edges, use_random_teleportation=False):
    """
    Generate the probability transition matrix for a graph with the given number of nodes and edges.
    Input:
        nodes_count: Number of nodes in the graph.
        edges: List of edges in the graph as a list of tuples of two vertices.
        use_random_teleportation: If True, the probability of teleporting to a random node (0.1) is used.
    Output:
        Probability transition matrix for the graph.
    """
    # Initialize with zeros
    probability_transition_matrix = np.zeros((nodes_count, nodes_count))
    for edge in edges:
        # Add one for each edge
        probability_transition_matrix[edge[0]][edge[1]] = 1
    # Get the number of 1s in each row
    row_sum = np.sum(probability_transition_matrix, axis=1).reshape((-1, 1))
    # Divide each row by the number of 1s in that row
    probability_transition_matrix = probability_transition_matrix / row_sum
    if not use_random_teleportation:
        # If we are not using random teleportation, this is it
        return probability_transition_matrix
    # If we are using random teleportation, we need to add the teleportation probability to each row
    # Multiplying the matrix with (1 - alpha)
    probability_transition_matrix = probability_transition_matrix * (1 - ALPHA)
    # Remaining sums will be the teleportation probabilities for each row
    remaining_sums = np.ones((nodes_count, 1)) * ALPHA
    remaining_sums = remaining_sums / (nodes_count - row_sum)
    # Complement will be used to add the teleportation probabilities to each row
    complement = np.zeros((nodes_count, nodes_count))
    complement[probability_transition_matrix == 0] = 1
    complement *= remaining_sums
    probability_transition_matrix += complement
    # Return the final matrix
    return probability_transition_matrix


def get_left_principal_eigenvector(probability_transition_matrix):
    """
    Gets the left principal eigenvector of the given probability transition matrix.
    This uses the numpy function to get the eigenvector directly.
    Input:
        probability_transition_matrix: Probability transition matrix for the graph.
    Output:
        Left principal eigenvector of the matrix.
    """
    # Get the eigenvalues and eigenvectors of the matrix
    v, V = np.linalg.eig(probability_transition_matrix.T)
    print(v)
    print(V)
    left_vec = V[:, 0].T
    left_vec = V[:, v.argmax()]
    left_vec = left_vec / sum(left_vec)
    left_vec = np.reshape(left_vec, (1, -1))
    # Return the principal left eigenvector
    return left_vec


def get_left_principal_eigenvector_power_iteration(probability_transition_matrix, nodes_count):
    """
    Gets the left principal eigenvector of the given probability transition matrix.
    This uses the power iteration method to obtain the eigenvector.
    Input:
        probability_transition_matrix: Probability transition matrix for the graph.
        nodes_count: Number of nodes in the graph.
    Output:
        Left principal eigenvector of the matrix.
    """
    # Initialize the vector with 0s
    first = np.full((1, nodes_count), 1 / nodes_count)
    prev = first
    # Set the first element to 1 (we are first at webpage 0)
    for i in range(1000):
        # Get the next vector
        prev = first
        first = np.dot(first, probability_transition_matrix)
        sum = np.sum((prev - first) ** 2)
        if sum < BREAK_THRESHOLD:
            break
    # Return the final vector
    return first

def main():
    """
    Main function.
    """
    arguments, values = getopt.getopt(sys.argv[1: ], 'r', 'use-random-teleportation')
    use_random_teleportation = False

    for argument, value in arguments:
        if argument == '--use-random-teleportation' or argument == '-r':
            print('Using random teleportation')
            use_random_teleportation = True

    # Get the inputs
    nodes_count = int(input('Number of nodes: '))
    edge_count = int(input('Number of edges: '))

    edges = []

    # Get the edges
    print('Mention each edge as a space separated list of two vertices: ')
    for i in range(edge_count):
        inp = input()
        edges.append(tuple([int(edge) for edge in inp.split()]))

    # Generate the probability transition matrix using the function we defined
    probability_transition_matrix = generate_probability_transition_matrix(
        nodes_count, edges, use_random_teleportation=use_random_teleportation)
    
    print('\nThe final probabilty transition matrix will be: ')
    print(probability_transition_matrix)
    # Print the final left principal eigenvectors using the two methods
    print('Probabilities using Numpy Function: ')
    print(get_left_principal_eigenvector(probability_transition_matrix))
    print('Probabilities using Power Iteration: ')
    ans = get_left_principal_eigenvector_power_iteration(probability_transition_matrix, nodes_count)
    prob = []
    for i in range(len(ans[0])):
        prob.append([i, ans[0][i]])
    top3prob = prob
    top3prob = sorted(top3prob, key=lambda x: x[1], reverse=True)
    top3prob = top3prob[0:3]
    print("Top 3 probabilities are:")
    print(tabulate(top3prob, headers=["Node", "Probability"]))
    print("\nList of all probabilities:")
    print(tabulate(prob, headers=["Node", "Probability"]))


if __name__ == '__main__':
    main()
