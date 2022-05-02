import networkx as nx
import re

import numpy as np
from tabulate import tabulate
from nltk.corpus import stopwords

web_graph = nx.read_gpickle('web_graph.gpickle')


def generate_sets(query_word, postings_list):
    """
    Generates the root and base sets for the query word
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
    adj = np.zeros((len(base_set), len(base_set)))
    bslist = list(base_set)
    for edge in web_graph.edges:
        if edge[0] in base_set and edge[1] in base_set:
            adj[bslist.index(edge[0])][bslist.index(edge[1])] = 1
    return adj


def generate_hub_authority_scores(adj_matrix):
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
    aTa = np.dot(adj_matrix.T, adj_matrix)
    aaT = np.dot(adj_matrix, adj_matrix.T)
    a_vec = np.full((1, len(adj_matrix)), 1/len(adj_matrix))
    h_vec = np.full((1, len(adj_matrix)), 1/len(adj_matrix))
    for i in range(100):
        a_vec = np.dot(a_vec, aTa)
        a_vec = a_vec / np.max(a_vec)
        h_vec = np.dot(h_vec, aaT)
        h_vec = h_vec / np.max(h_vec)
    a_vec = a_vec/sum(a_vec[0])
    h_vec = h_vec/sum(h_vec[0])
    return a_vec, h_vec


web_graph = nx.read_gpickle('web_graph.gpickle')
# Create a postings list
postings_list = {}
stop_words = set(stopwords.words('english'))
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

query = input("Enter query word: ")
query = query.split()
query = query[0]
rs, bs = generate_sets(query, postings_list)
adjacency_list = generate_adj_matrix(bs)

authority, hub = generate_hub_authority_scores_power_iteration(adjacency_list)
print(authority)
base_set_list = list(bs)
scores = []
auth_sum = 0
hub_sum = 0
for i in range(len(authority[0])):
    if authority[0][i] <= 0:
        authority[0][i] = authority[0][i] * -1
    if hub[0][i] < 0:
        hub[0][i] = hub[0][i] * -1
    scores.append([base_set_list[i], authority[0][i], hub[0][i]])
    auth_sum = auth_sum + authority[0][i]
    hub_sum = hub_sum + hub[0][i]

print(tabulate(scores, headers=["Node", "Authority Score", "Hub Score"]))
print("Authority score sum = " + str(auth_sum))
print("Hub score sum = " + str(hub_sum))
