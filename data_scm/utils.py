import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
from collections import Counter
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

## Util sampler functions
trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : round(np.random.uniform(a, b))

def causes_sampler_f(num_causes, min_lb, max_lb, max_len):
    assert(max_lb-min_lb>0)
    assert(max_lb+max_len<=1)
    lb = np.random.uniform(low=min_lb, high=max_lb, size=num_causes)
    ub = lb + np.random.uniform(low=0.0, high=max_len, size=num_causes)
    return lb, ub
            

def class_sampler_f(min_, max_):
    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2
    return s

## Find specific nodes in a graph
def get_all_descendants(graph, node):
    visited = set()
    stack = [node]

    descendants = set()

    while stack:
        current = stack.pop()
        for child in graph.successors(current):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                stack.append(child)
    return descendants

def get_all_antecedents(graph, node):
    visited = set()
    stack = [node]

    antecedents = set()

    while stack:
        current = stack.pop()
        for parent in graph.predecessors(current):
            if parent not in visited:
                visited.add(parent)
                antecedents.add(parent)
                stack.append(parent)
    return antecedents

def get_parents(graph, node):
    return set(graph.predecessors(node))

def get_grandparents(graph, node):
    grandparents = set()
    for parent in graph.predecessors(node):
        for grandparent in graph.predecessors(parent):
            grandparents.add(grandparent)
    return grandparents

def get_siblings(graph, node):
    siblings = set()
    for parent in graph.predecessors(node):
        for child in graph.successors(parent):
            if child != node:
                siblings.add(child)
    return siblings

def get_markov_blanket(G, node):
    """Return the Markov blanket of a node in a DAG."""
    # Parents
    parents = set(G.predecessors(node))
    
    # Children
    children = set(G.successors(node))
    
    # Co-parents (parents of the children, excluding the node itself)
    coparents = set()
    for child in children:
        coparents.update(G.predecessors(child))
    coparents.discard(node)
    
    # Markov blanket is union of all three
    markov_blanket = parents.union(children).union(coparents)
    
    return list(markov_blanket)

def get_non_markov_blanket(G, node):
    """Return all nodes that are NOT in the Markov blanket of `node` in a DAG."""
    # Get the Markov blanket
    mb = set(get_markov_blanket(G, node))
    
    # All nodes in the graph
    all_nodes = set(G.nodes)
    
    # Remove the node itself and its MB
    non_mb_nodes = all_nodes - mb - {node}
    
    return list(non_mb_nodes)

## Get assignment and generated data information
def get_info_assignment(assignment_list, target_node):
    info_assignment = {}
    
    # Target information
    activation_target = assignment_list[target_node]['assignment'].activation
    noise_std_target = assignment_list[target_node]['assignment'].noise_std
    info_assignment['target'] = {'activation':activation_target, 'noise_std':noise_std_target, 'parents':assignment_list[target_node]['parents']}

    # Direct cause information
    info_parents = {}
    for parent_node in assignment_list[target_node]['parents']:
        if parent_node in assignment_list.keys():
            info_parents[parent_node] = {'activation':assignment_list[parent_node]['assignment'].activation, 'noise_std':assignment_list[parent_node]['assignment'].noise_std}
    info_assignment['parents'] = info_parents

    # Statistics information
    activation_list = [str(assignment_list[node]['assignment'].activation) for node in assignment_list.keys()]
    
    ## Count the percentages of activations in the graph
    counter = Counter(activation_list)
    total = len(activation_list)
    percentages_activations = {key: (value / total) * 100 for key, value in counter.items()}

    ## Compute the mean noise std in the graph
    mean_noise_std = np.mean([assignment_list[node]['assignment'].noise_std.item() for node in assignment_list.keys()])

    info_assignment['statistics'] = {'percentage_activations':percentages_activations, 'mean_noise_std':mean_noise_std}

    return info_assignment

def get_info_discretization(target_continuous, target_discrete, class_boundaries):
    info_discretization = {}

    # # Compute boundrary quantiles
    # info_discretization['boundrary_quantiles'] = [(target_continuous <= boundary_i).float().mean().item() for boundary_i in class_boundaries]
    
    # Count class percentages
    vals, counts = torch.unique(target_discrete, return_counts=True)
    percentages = counts / len(target_discrete)
    percentages_class = {v.item(): p.item() for v, p in zip(vals, percentages)}
    info_discretization['percentage_class'] = percentages_class

    return info_discretization

def get_MIs(X, y, task):
    if task == 'classification':
        return mutual_info_classif(X.cpu().numpy(), y.cpu().numpy())
    elif task == 'regression':
        return mutual_info_regression(X.cpu().numpy(), y.cpu().numpy())

def get_info_MIs(G, selected_target, selected_features, data, discrete_y):
    parents = get_parents(G, selected_target)
    grandparents = get_grandparents(G, selected_target)
    siblings = get_siblings(G, selected_target)
    non_direct_cause = set(selected_features) - parents

    x_parents = torch.cat([data[node] for node in parents], dim=1)
    x_grandparents = torch.cat([data[node] for node in grandparents], dim=1)
    x_siblings = torch.cat([data[node] for node in siblings], dim=1)
    x_all = torch.cat([data[node] for node in selected_features], dim=1)
    x_non_direct_cause = torch.cat([data[node] for node in non_direct_cause], dim=1)

    y = data[selected_target].squeeze()

    info_MIs = {
        "MI_y_parents_continuous":{k:v for k,v in zip(parents, get_MIs(x_parents, y, 'regression'))},
        "MI_y_grandparents_continuous":{k:v for k,v in zip(grandparents, get_MIs(x_grandparents, y, 'regression'))},
        "MI_y_siblings_continuous":{k:v for k,v in zip(siblings, get_MIs(x_siblings, y, 'regression'))},
        "MI_y_all_continuous_mean":np.mean(get_MIs(x_all, y, 'regression')),
        "MI_discrete_y_parents":{k:v for k,v in zip(parents, get_MIs(x_parents, discrete_y, 'classification'))},
        "MI_discrete_y_grandparents":{k:v for k,v in zip(grandparents, get_MIs(x_grandparents, discrete_y, 'classification'))},
        "MI_discrete_y_siblings":{k:v for k,v in zip(siblings, get_MIs(x_siblings, discrete_y, 'classification'))},
        "MI_discrete_y_all_mean":np.mean(get_MIs(x_all, discrete_y, 'classification')),
        "MI_discrete_y_direct_cause_mean": np.mean(get_MIs(x_parents, discrete_y, 'classification')),
        "MI_discrete_y_non_direct_cause": np.mean(get_MIs(x_non_direct_cause, discrete_y, 'classification'))
    
    }
    return info_MIs

## Visualization of the DAG
def draw_graph(G, path):
    plt.figure(figsize=(8, 6))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray', arrows=True)
    plt.savefig(path)
    plt.show()

def convert_categorical_type(data):
    for col in data.columns:
        if len(data[col].unique())<2000:
            data[col] = data[col].astype("category")
    
    return data

def random_acyclic_orientation(G_und):
    """
    Take an igraph undirected graph and return a DAG (as a NetworkX DiGraph)
    by orienting edges according to a random node ordering.
    """
    n = G_und.vcount()
    nodes = list(range(n))

    # Random topological order
    random.shuffle(nodes)
    order = {node: i for i, node in enumerate(nodes)}

    # Build DAG in networkx
    G_dag = nx.DiGraph()
    G_dag.add_nodes_from(nodes)

    for u, v in G_und.get_edgelist():
        if order[u] < order[v]:
            G_dag.add_edge(u, v)
        else:
            G_dag.add_edge(v, u)

    return G_dag