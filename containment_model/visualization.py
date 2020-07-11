import networkx as nx
import matplotlib.pyplot as plt
import collections
import numpy as np

def plot_network(network):
    plt.figure(figsize = (10, 10))
    nx.drawing.draw_networkx_edges(network.g, pos=network.posdict, alpha=0.1)
    plt.xlim(0, network.size)
    plt.ylim(0, network.size)
def plot_degree(network, ax = None):
    if ax is None:
        plt.figure(figsize = (10, 10))
        ax = plt.gca()
    degree_sequence = sorted([d for n, d in network.g.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    #plt.figure()
    ax.bar(deg, cnt, color='b')

    #ax.title("Degree Histogram")
    ax.set_ylabel("Count")
    ax.set_xlabel("Degree")
    #ax.set_xticks([d + 0.4 for d in deg])
    #ax.set_xticklabels(deg)
def plot_edge_length(network):
    """
    Plot the distribution of edge lengths
    """
    plt.figure(figsize = (10, 10))
    diff = network.pos[network.edges[:,0]] - network.pos[network.edges[:,1]]
    length = np.linalg.norm(diff, axis=1)
    plt.hist(length, bins=100, range=(0, np.percentile(length, 90) * 2))

def states_stackplot(model, t = None):
    if t is None:
        t = model.t
    state = model.region_history.groupby(['t','state']).total.sum().to_numpy().reshape(-1, 4)
    state = state / len(model.nodes)
    plt.stackplot(np.arange(t+1), state[:t+1].T, colors=['tab:green', 'tab:orange', 'tab:red', 'tab:blue'])
    plt.xlim(0, t)
    plt.ylim(0, 1)

def states_scatter(model, t = None):
    if t is None:
        t = model.t
    plt.figure(figsize = (10, 10))
    states = [
        ('susceptible', 0, 'tab:green'),
        ('exposed', 1, 'tab:orange'),
        ('infected', 2, 'tab:red'),
        ('recovered', 3, 'tab:blue')
    ]
    nodes = model.state_history.xs(t, level=('t'))
    for (state, i, color) in states:
        state_nodes = nodes[nodes.state == i]
        plt.scatter(state_nodes.x, state_nodes.y, c=color, s=3)