from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
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


class GridRegionClassifier:
    def __init__(self, size, grid_dim):
        self.grid_dim = grid_dim
        self.n  = grid_dim ** 2
        self.size = size
        self.region_size = size / grid_dim
        self.reg_bins = np.linspace(0, size, grid_dim + 1)
    def classify(self, x, y):
        x_reg = np.digitize(x, self.reg_bins) - 1
        y_reg = np.digitize(y, self.reg_bins) - 1
        node_region = y_reg * self.grid_dim + x_reg
        return node_region
    def draw(self, ax, regions):
        pass

class OutbreakModel:
    def __init__(self, graph, region_classifier, config):
        self.config = config
        self.region_classifier = region_classifier
        self.graph = graph.to_directed()
        self.region_index = pd.RangeIndex(region_classifier.n, name='region')
        self.node_index = pd.RangeIndex(len(self.graph), name='node')
        self.edge_index = pd.RangeIndex(self.graph.size(), name='edge')
        self.state_index = pd.Index([0, 1, 2, 3], name='state')
        
        # Nodes
        nodes_list = [(d['pos'][0], d['pos'][1], 0, 0, 0, 0) for n,d in graph.nodes(data=True)]
        self.nodes = pd.DataFrame(
            nodes_list,
            columns = ['x', 'y', 'state', 'counter', 'region', 'new_state'],
            index = self.node_index
        )
        self.nodes.region = self.region_classifier.classify(self.nodes.x, self.nodes.y)
        
        # Edges
        edgelist = nx.to_pandas_edgelist(self.graph)

        self.edges = pd.DataFrame(
            {
                'source' : edgelist.source,
                'target' : edgelist.target,
                'source_region' : None,
                'target_region' : None,
                'weight' : 1.0
            },
            index = self.edge_index
        )
        
        self.edges.source_region = pd.merge(self.edges, self.nodes, left_on='source', right_index=True).region
        self.edges.target_region = pd.merge(self.edges, self.nodes, left_on='source', right_index=True).region
        # Regions
        self.regions = pd.DataFrame(
            self.nodes.groupby('region').size().astype('int'),
            columns = ['total_nodes'],
            index = self.region_index
        )
        
        # Transmission suppression
        self.suppression = pd.DataFrame(
            columns=['factor'],
            index = pd.MultiIndex.from_product(
                [self.region_index, self.region_index],
                names = ['source_region', 'region']
            )
        )
        
        # State counts by region
        self.state_counts = pd.DataFrame(
            columns = ['total', 'fraction', 'new'],
            index = pd.MultiIndex.from_product(
                [self.region_index, self.state_index],
                names = ['region', 'state']
            )
        )
        
        self.reset()
        
    def set_region_counts(self):
        # Update state counts
        new_counts = self.nodes.groupby('region').state.value_counts()
        new_counts_aligned, _ = new_counts.align(self.state_counts, fill_value = 0)
        self.state_counts.total = new_counts_aligned
        self.state_counts.fraction = self.state_counts.total / self.regions.total_nodes
    
    def set_suppression(self):
        lockdown_threshold = self.config['infection_rate']
        suppression = self.config['lockdown_suppression']
        # Update suppressions
        lockdown = self.state_counts.xs(2, level='state')\
            .fraction.gt(lockdown_threshold)\
            .reindex(self.suppression.index, level='region')
        self.suppression.factor = 1.0 - suppression * lockdown
        # TODO: move this to a separate class and implement more complex
        # responses
    
    def rate_transition(self, initial, final, rate):
        transition_candidates = self.nodes[self.nodes.state == initial].index
        p = np.random.uniform(size = len(transition_candidates))
        self.nodes.loc[transition_candidates[p < rate], 'new_state'] = final
    '''
    Perform one step of the transition
    New 
    '''
    def transition_step(self):
        # Set new states to old states
        self.nodes.new_state = self.nodes.state
        
        # susceptible -> exposed
        # Select transmission edges
        source_state = pd.merge(self.edges, self.nodes, left_on='source', right_index=True).state
        target_state = pd.merge(self.edges, self.nodes, left_on='target', right_index=True).state
        
        active_edges = self.edges[
            #((source_state == 1) | (source_state == 2)) & (target_state == 0)
            (source_state == 2) & (target_state == 0)
        ]
        active_edges = pd.merge(
            active_edges, self.suppression, 
            left_on=['source_region', 'target_region'], 
            right_on=['source_region', 'region'],
            copy=False
        )
    
        # Select the transmissions
        p = np.random.uniform(size = len(active_edges))
        infection_rate = self.config['infection_rate']
        transmissions = active_edges[p < active_edges.factor * active_edges.weight * infection_rate]
        self.nodes.loc[transmissions.target, 'new_state'] = 1
        
        # exposed -> infected
        exposed_transition_rate = self.config['exposed_transition_rate']
        self.rate_transition(1, 2, exposed_transition_rate)
        
        # infected -> recovered
        recovery_rate = self.config['recovery_rate']
        self.rate_transition(2, 3, recovery_rate)
        
        # recovered -> susceptible
        deimmunization_rate = self.config['deimmunization_rate']
        self.rate_transition(3, 0, deimmunization_rate)
        
        # Set the state to the new state
        self.nodes.state = self.nodes.new_state
        self.t += 1
    '''
    Write the current state into the output at time t. 
    '''
    def write_state_output(self, t):
        self.state_history.loc[(t, slice(None))] = \
            self.nodes.set_index(self.state_history_index(1, t))
    
        self.region_history.loc[(t, slice(None), slice(None))] = \
            self.state_counts.set_index(self.region_history_index(1, t))
        
    '''
    Perform one iteration
    '''
    def iteration(self):
        self.set_suppression()
        self.transition_step()
        self.set_region_counts()
        self.write_state_output(self.t)
    
    def region_history_index(self, size, start = 0):
        new_index = pd.MultiIndex.from_product(
            [pd.RangeIndex(start, start + size), self.region_index, self.state_index], 
            names=['t', 'region', 'state']
        )
        return new_index
    
    def state_history_index(self, size, start = 0):
        new_index = pd.MultiIndex.from_product(
            [pd.RangeIndex(start, start + size), self.node_index], 
            names=['t', 'node']
        )
        return new_index
    
    def make_region_history(self, size, offset = 0):
        new_region_history = pd.DataFrame(
            0, 
            index = self.region_history_index(size, offset), 
            columns = self.state_counts.columns
        ).astype(self.state_counts.dtypes)
        return new_region_history
    def make_state_history(self, size, offset = 0):
        new_state_history = pd.DataFrame(
            0, 
            index = self.state_history_index(size, offset), 
            columns = self.nodes.columns
        ).astype(self.nodes.dtypes)
        return new_state_history
    def resize_output(self, size):
        # Create array for the region history      
        if (size > self.output_size):
            self.state_history = self.state_history.append(
                self.make_state_history(size - self.output_size, self.output_size),
            )
            self.region_history = self.region_history.append(
                self.make_region_history(size - self.output_size, self.output_size),
            )
            self.output_size = size

    '''
    Clear the output and set the 
    '''
    def reset(self, size = 1):
        self.t = 0
        self.output_size = size
        self.state_history = pd.DataFrame(
            0, 
            index = self.state_history_index(size), 
            columns = self.nodes.columns
        ).astype(self.nodes.dtypes)
        self.region_history = pd.DataFrame(
            0, 
            index = self.region_history_index(size), 
            columns = self.state_counts.columns
        ).astype(self.state_counts.dtypes)
        
        self.set_region_counts()
        self.write_state_output(0)
    
    '''
    Run the diffusion for n steps. 
    '''
    def run(self, n):
        self.resize_output(self.t + n + 1)
        for i in tqdm(range(self.t, self.t + n)):
            self.iteration()
    def save_frame(self, t):
        fig = plt.figure(figsize=(10, 10))
        snapshot = self.state_history.xs(t, level='t')
        plt.scatter(snapshot.x, snapshot.y, s=2, c=snapshot.state)
        plt.savefig("frame%d.png" % t)
        plt.close(fig)