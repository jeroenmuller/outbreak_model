import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from copy import copy
import random
import os
from tqdm import tqdm # progress bar
from matplotlib.patches import Rectangle

from .utils import classify_grid
class OutbreakModel:
    def __init__(self, network, config):
        self.config = config
        self.network = network
        self.graph = network.g
        self.num_regions = config['regions'][0] * config['regions'][1] 
        self.region_index = pd.RangeIndex(self.num_regions, name='region')
        self.node_index = pd.RangeIndex(network.pos.shape[0], name='node')
        self.edge_index = pd.RangeIndex(self.graph.size() * 2, name='edge')
        self.state_index = pd.Index([0, 1, 2, 3], name='state')
        
        # Nodes
        self.nodes = pd.DataFrame(
            {
                'x'         : self.network.pos[:,0],
                'y'         : self.network.pos[:,1],
                'state'     : 0,
                'counter'   : 0,
                'region'    : 0,
                'new_state' : 0
            },
            index = self.node_index
        )

        center = self.network.size / 2
        self.nodes['r'] = np.sqrt((self.nodes.x - center)**2 + (self.nodes.y - center)**2)

        self.nodes['region'] = classify_grid(
            self.nodes.x, 
            self.nodes.y, 
            (self.network.size, self.network.size),
            self.config['regions']
        )
        
        # Edges
        edgelist = nx.to_pandas_edgelist(self.graph.to_directed())

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
            columns=['factor', 'countdown'],
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
        lockdown_threshold = self.config['lockdown_threshold']
        suppression = self.config['lockdown_suppression']
        lockdown_time = self.config['lockdown_time']
        # Update suppressions
        lockdown_trigger = self.state_counts.xs(2, level='state')\
            .fraction.gt(lockdown_threshold)\
            .reindex(self.suppression.index, level='region')
        print('Lockdown trigger in %d regions' % sum(lockdown_trigger))
        #print(self.suppression.loc[lockdown_trigger, 'countdown'])
        self.suppression.loc[self.suppression.countdown > 0, 'countdown'] -= 1
        self.suppression.loc[lockdown_trigger, 'countdown'] = lockdown_time
        lockdown = (self.suppression.countdown != 0)
        print('Lockdown in %d regions' % sum(lockdown))
        #print(self.suppression)
        self.suppression.factor = 1.0 - suppression * lockdown

        # TODO: move this to a separate class and implement more complex
        # responses
    
    def rate_transition(self, initial, final, rate):
        transition_candidates = self.nodes[self.nodes.state == initial].index
        p = np.random.uniform(size = len(transition_candidates))
        self.nodes.loc[transition_candidates[p < rate], 'new_state'] = final
    
    def time_transition(self, initial, final, time):
        transition_candidates = self.nodes[
        (self.nodes.state == initial) & (self.nodes.counter > time)].index
        self.nodes.loc[transition_candidates, 'new_state'] = final

    '''
    Perform one step of the transition
    New 
    '''
    def transition_step(self):
        # Set new states to old states
        self.nodes.counter += 1
        self.nodes.new_state = self.nodes.state
        

        # susceptible -> exposed
        # Select transmission edges
        source_state = pd.merge(self.edges, self.nodes, left_on='source', right_index=True).state
        target_state = pd.merge(self.edges, self.nodes, left_on='target', right_index=True).state
        
        active_edges = self.edges.loc[
            #((source_state == 1) | (source_state == 2)) & (target_state == 0)
            (source_state == 2) & (target_state == 0)
        ]

        #print("%d infections lead to %d active edges out of %d total" % (np.sum(self.nodes.state == 2), len(active_edges), len(self.edges)))

        active_edges = pd.merge(
            active_edges, self.suppression, 
            left_on=['source_region', 'target_region'], 
            right_on=['source_region', 'region'],
            copy=False
        )
    
        # Select the transmissions
        p = np.random.uniform(size = len(active_edges))
        r = self.config['transmission_coefficient']
        deg = self.network.config['degree']
        infected_time = self.config['infected_time']
        infection_rate = r / (deg * infected_time)
        transmissions = active_edges[p < active_edges.factor * active_edges.weight * infection_rate]
        
        #print("%d cases infected %d out of %d" % (np.sum(self.nodes.state == 2), len(transmissions), len(active_edges)))
        self.nodes.loc[transmissions.target, 'new_state'] = 1
        
        # exposed -> infected
        exposed_transition_time = self.config['exposed_time']
        #self.rate_transition(1, 2, exposed_transition_rate)
        self.time_transition(1, 2, exposed_transition_time)
        
        # infected -> recovered
        recovery_time = self.config['infected_time']
        #self.rate_transition(2, 3, recovery_rate)
        self.time_transition(2, 3, recovery_time)
        
        # recovered -> susceptible
        #deimmunization_rate = self.config['deimmunization_rate']
        #self.rate_transition(3, 0, deimmunization_rate)
        immune_time = self.config['immune_time']
        self.rate_transition(3, 0, immune_time)
        
        # Reset counter
        self.nodes.loc[self.nodes.state != self.nodes.new_state, 'counter'] = 0

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
        self.suppression.factor = 1.0
        self.suppression.countdown = 0
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
        self.initialize_states()
        self.set_region_counts()
        self.write_state_output(0)

    def initialize_states(self):
        radius = self.config.get('outbreak_start_radius')
        if not radius is None:
            initial = (self.nodes.r < radius)
            self.nodes.loc[initial, 'state'] = 1
            random = np.random.randint(self.config['exposed_time'], size=len(self.nodes))
            self.nodes.counter.mask(initial, random, inplace=True)


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