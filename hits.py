import networkx as nx
import re

import numpy as np
from tabulate import tabulate
from nltk.corpus import stopwords

web_graph = nx.read_gpickle('web_graph.gpickle')


def generate_sets(query_word, postings_list):
    """
    Generates the root and base sets for the query word

    Takes the entire posting list of the query word as root set.
    Initialises the base set with root set.
    Then iterates through all the edges in the given graph. If at least one of them is in the root set,
    it adds the other to the base set.
    """
    if query_word not in postings_list:
        raise Exception('Word not in the postings list')
    root_set = postings_list[query_word]
    base_set = []
    for node in root_set:
        base_set.append(node)
    root_set = set(root_set)
    base_set = set(base_set)
    for edge in web_graph.edges:
        if edge[0] in root_set:
            base_set.add(edge[1])
        if edge[1] in root_set:
            base_set.add(edge[0])
    return root_set, base_set


def generate_adj_matrix(base_set):
    """
    Generates the adjacency matrix.
    Takes the base set, then iterates through all the edges in the edge set.
    If both endpoints of some edge (a,b) are present in base set, it sets
    adj[a][b] to 1, indicating that there's an edge from a to b.
    """
    adj = np.zeros((len(base_set), len(base_set)))
    bslist = list(base_set)
    for edge in web_graph.edges:
        if edge[0] in base_set and edge[1] in base_set:
            adj[bslist.index(edge[0])][bslist.index(edge[1])] = 1
    return adj


def generate_hub_authority_scores(adj_matrix):
    """
    Generates the hub and authority scores.
    It takes the adjacency matrix and then finds the left eigenvector directly using numpy, i.e. the
    eigenvector corresponding to the largest eigenvalue, of (AT)A to find the authority scores,
    and of A(AT) to find the hub scores. Here A denotes the adjacency matrix, and AT denotes its transpose.

    """
    aTa = np.dot(adj_matrix.T, adj_matrix)
    aaT = np.dot(adj_matrix, adj_matrix.T)
    v, V = np.linalg.eig(aaT.T)
    left_vec = V[:, 0].T
    left_vec = V[:, v.argmax()]
    left_vec = left_vec / sum(left_vec)
    h_vec = np.reshape(left_vec, (1, -1))
    v, V = np.linalg.eig(aTa.T)
    left_vec = V[:, 0].T
    left_vec = V[:, v.argmax()]
    left_vec = left_vec / sum(left_vec)
    a_vec = np.reshape(left_vec, (1, -1))
    # Return the principal left eigenvector
    return a_vec, h_vec


def generate_hub_authority_scores_power_iteration(adj_matrix):
    """
    Generates the hub and authority scores.
    It takes the adjacency matrix and then finds the left eigenvector using the power iteration method, i.e. the
    eigenvector corresponding to the largest eigenvalue of (AT)A to find the authority scores,
    and of A(AT) to find the hub scores. Here A denotes the adjacency matrix, and AT denotes its transpose.

    For this, we initialise our authority and hub scores with 1/n, where n is the number of nodes.
    Then, we multiply authority and hub scores by aTa and aaT respectively, and normalise the result
    by dividing it by the largest value in the matrix.
    We do these iterations until the change after iteration becomes smaller than some threshold value.
    However, in this implementation we have run the iteration 1000 times, as this generally guarantees convergence.
    Then we divide the resulting scores with the sum of all scores, to bring them in the range [0,1], and then
    return them.
    """
    aTa = np.dot(adj_matrix.T, adj_matrix)
    aaT = np.dot(adj_matrix, adj_matrix.T)
    a_vec = np.full((1, len(adj_matrix)), 1/len(adj_matrix))
    h_vec = np.full((1, len(adj_matrix)), 1/len(adj_matrix))
    for i in range(1000):
        a_vec = np.dot(a_vec, aTa)
        a_vec = a_vec / np.max(a_vec)
        h_vec = np.dot(h_vec, aaT)
        h_vec = h_vec / np.max(h_vec)
    a_vec = a_vec/sum(a_vec[0])
    h_vec = h_vec/sum(h_vec[0])
    return a_vec, h_vec


"""
Reading the dataset from web_graph.gpickle
"""
web_graph = nx.read_gpickle('web_graph.gpickle')
# Create a postings list
postings_list = {}
stop_words = set(stopwords.words('english'))

"""
Creating a posting list by iterating through all nodes, then doing some pre processing like
removing non-alphabetical characters, removing multiple spaces, and converting to lowercase.
Then we filter out all the stop words, and creating posting list for the other words.
"""
for i in range(len(web_graph.nodes)):
    node = web_graph.nodes[i]
    content = node['page_content']
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.lower()
    content = [c for c in content.split(' ') if c != '' and len(c) > 2]
    for word in content:
        if word not in stop_words:
            if word not in postings_list:
                postings_list[word] = [i]
            elif postings_list[word][-1] != i:
                postings_list[word].append(i)

"""
Taking input from user.
Then we generate the root and base sets for the given query term.
Then we generate the adjacency matrix using the obtained base set, and
use the adjacency matrix to find authority and hub values.
Finally we print the obtained authority and hub values in a tabulated format.
"""
query = input("Enter query word: ")
query = query.split()
query = query[0]
rs, bs = generate_sets(query, postings_list)
adjacency_list = generate_adj_matrix(bs)

authority, hub = generate_hub_authority_scores_power_iteration(adjacency_list)
base_set_list = list(bs)
scores = []
scores_auth = []
scores_hub = []
auth_sum = 0
hub_sum = 0
for i in range(len(authority[0])):
    if authority[0][i] <= 0:
        authority[0][i] = authority[0][i]*-1
    if hub[0][i] < 0:
        hub[0][i] = hub[0][i]*-1
    scores.append([base_set_list[i],authority[0][i],hub[0][i]])
    scores_auth.append([base_set_list[i],authority[0][i]])
    scores_hub.append([base_set_list[i],hub[0][i]])
    auth_sum = auth_sum+authority[0][i]
    hub_sum = hub_sum+hub[0][i]

scores_auth = sorted(scores_auth, key=lambda x:x[1], reverse=True)
scores_auth = scores_auth[0:3]
scores_hub = sorted(scores_hub, key=lambda x:x[1],reverse=True)
scores_hub = scores_hub[0:3]

print("Top 3 authority scores are:")
print(tabulate(scores_auth, headers = ["Node", "Authority Score"]))
print("\nTop 3 Hub scores are:")
print(tabulate(scores_hub, headers = ["Node", "Hub Score"]))

print("\n List of all scores:")
print(tabulate(scores, headers = ["Node", "Authority Score", "Hub Score"]))
print("Authority score sum = "+str(auth_sum))
print("Hub score sum = "+str(hub_sum))
