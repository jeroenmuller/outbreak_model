from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from networkx.generators.geometric import random_geometric_graph, soft_random_geometric_graph, waxman_graph
from copy import copy

import collections
import random
from datetime import datetime
import os

from subprocess import call
from tqdm import tqdm # progress bar

from matplotlib.patches import Rectangle
from .utils import distance_kernel, sample_pairwise_dist
'''
Sample X according to the density function, which is a function of the centers
'''
class InteractionNetwork:
    def __init__(self, config):
        # Linear size of the simulation domain
        self.config = config
        self.size = config['size']
        self.area = self.size ** 2
        self.N = config['population_density'] * self.area
        self.N_centers = int(config['cluster_density'] * self.area)
        self.generate()

    def density(self, centers, X):
        min_distance = np.min(cdist(self.centers, X), axis=0)
        cluster_radius = self.config['cluster_radius']
        background_density = self.config['background_density']
        return distance_kernel(min_distance, cluster_radius, p=2) + background_density
        #return np.exp(-min_distance / cluster_radius) + background_density

    def travel_kernel(self, X):
        return self.config['link_probability'] / (1. + (X / (0.5 / self.config['travel_radius'])) ** 3.8)
    def generate(self):
        # Generate the centers
        self.centers = np.random.uniform(size=(self.N_centers, 2)) * self.size
        

        self.pos = self.sample_density(self.N)
        self.edges = geometric_graph_degree(self.pos, self.config['travel_radius'], self.config['degree'], connect_isolated=True)

        # Create a dict of the coordinates
        self.posdict = dict(enumerate(self.pos.tolist()))
        
        # Create the connection graph
        self.g = nx.Graph()
        self.g.add_edges_from(self.edges.tolist())
        nx.set_node_attributes(self.g, self.posdict, 'pos')
        #g = soft_random_geometric_graph(self.N, 2 * self.config['travel_radius'], pos=self.posdict, p_dist = self.travel_kernel)
        print("Number of centers: {}\nNumber of nodes: {}\nNumber of links: {}".format(self.N_centers, self.N, len(self.g.edges)))
        
    def sample_density(self, N, batch_size = 1000):
        max_density = np.max(self.density(self.centers, self.centers))
        out = np.zeros(shape=(N, 2))
        out_filled_end = 0
        while (out_filled_end < N):
            batch = np.random.uniform(size=(batch_size, 3))
            
            ## Scale the candidate samples to match the area size
            batch[:,:2] *= self.size
            batch_density = self.density(self.centers, batch[:,:2])
            # Which samples to keep
            keep = np.argwhere((batch[:,2]  / max_density) < batch_density).flatten()
            keep = keep[:N - out_filled_end]

            out[out_filled_end : out_filled_end + len(keep)] = batch[keep][:,:2]
            out_filled_end += len(keep)
        return out

from scipy.spatial.distance import cdist, pdist
from scipy.spatial import cKDTree


'''
Sample a soft gemoetric graph with the specified degree
'''
def geometric_graph_degree(pos, r, degree, connect_isolated = True):
    area = (np.max(pos[:,0]) - np.min(pos[:,0])) * (np.max(pos[:,1]) - np.min(pos[:,1]))
    N = len(pos)
    target_pairs = (degree * N // 2)
    density = N / area
    p = 10 * degree / (np.pi * r**2 * density)
    pairs = []
    while len(pairs) < target_pairs:
        pairs = sample_pairwise_dist(pos, r, p)
        p *= 2
        if p > 2:
            break
    if (len(pairs) < target_pairs):
        print("Density too low to reach desired graph degree with specified radius")
        return None
      
    ## Sample pairs to get the required mean degree
    seleciton = np.random.choice(len(pairs), size=target_pairs, replace=False)
    selected_pairs = pairs[seleciton]
    
    if connect_isolated:
        # Find nodes that have no edges
        is_isolated = ~np.isin(np.arange(N), selected_pairs.flat)
        isolated_nodes = np.nonzero(is_isolated)[0]

        # Add edges to the nearest neighbour
        connected_kd = cKDTree(pos[~is_isolated])
        nearest = connected_kd.query(pos[is_isolated])[1]
        nearest_idx = np.arange(len(pos))[~is_isolated]
        isolated_links = np.array([isolated_nodes, nearest_idx[nearest]]).T
        selected_pairs = np.concatenate([
            selected_pairs[:len(selected_pairs) - len(isolated_links)], 
            isolated_links
        ])
       
    return selected_pairs