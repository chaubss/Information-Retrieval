import numpy as np
import time

ALPHA = 0.1


def generate_probability_transition_matrix(nodes_count, edges, use_random_teleportation=True):
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
    left_vec = V[:, 0].T
    left_vec = V[:, v.argmax()]
    left_vec = left_vec / sum(left_vec)
    left_vec = np.reshape(left_vec, (1, -1))
    # Return the principal left eigenvector
    return left_vec

def main():
    """
    Main function.
    """
    # Get the inputs
    # nodes_count = int(input('Number of nodes: '))
    # edge_count = int(input('Number of edges: '))
    nodes_count = int(input())
    # edge_count = int(input())
    
    edges = []
    for i in range(nodes_count - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    edge_count = len(edges)

    print('Number of edges:', edge_count)

    start = time.time()
    # Generate the probability transition matrix using the function we defined
    probability_transition_matrix = generate_probability_transition_matrix(
        nodes_count, edges)

    # Print the final left principal eigenvectors using the two methods
    asdfsdf = get_left_principal_eigenvector(probability_transition_matrix)
    end = time.time()
    print('Time taken: ', end - start)

if __name__ == '__main__':
    main()
