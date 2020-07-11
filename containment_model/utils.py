import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from tqdm import tqdm

def classify_grid(x, y, domain_size, cells):
    """
    Classify the coordinates (x[i], y[i]) into equally sized rectangular regions.

    Keyword arguments:
    x             -- x coordinate
    y             -- y coordinate
    domain_size   -- (size_x, size_y)
    cells         -- (num_cells_x, num_cells_y)
    """
    bins_x = np.linspace(0, domain_size[0], cells[0] + 1)
    bins_y = np.linspace(0, domain_size[1], cells[1] + 1)

    x_cell = np.digitize(x, bins_x) - 1
    y_cell = np.digitize(y, bins_y) - 1
    cell = y_cell * cells[0] + x_cell
    return cell

def rejection_sample():
    """
    Reje
    """
    pass

def sample_pairwise_dist(pos, radius, p, kernel = None, max_batch_size = 100000, min_binsize = 0.4):
    """
    Sample pairwise distance pdf to construct a soft geometric random graph
    We want to use cdist because it is much faster and more stable than kdTree.

    Keyword arguments:
    pos         -- an (n x 2) ndarray containing the coordinates of the points we want 
                   to connect
    radius      -- Maximum distance for connecting two points
    p           -- Scaling factor for the probability of keeping a link
    max_batch_size -- Maximum size of cdist in the inner loop. Larger values run 
                      faster but require more memory
    min_binsize -- Minimum (spatial) size of bins. Smaller bins reduce the required 
                   number of distance computations but increase overhead. 

    Returns:
    An (n x 2) ndarray of ints, where each entry represents a link between the corresponding indices 
    in the pos array. 
    """
    pairs = []
    min_x = np.min(pos[:,0])
    max_x = np.max(pos[:,0])
    min_y = np.min(pos[:,1])
    max_y = np.max(pos[:,1])
    
    binsize = max(radius, min_binsize)
    
    bins_x = np.arange(min_x, max_x + binsize, binsize)
    bins_y = np.arange(min_y, max_y + binsize, binsize)
    
    x = np.digitize(pos[:,0], bins_x)
    y = np.digitize(pos[:,1], bins_y)
    
    bins = np.zeros(shape=(len(bins_x), len(bins_y)), dtype=object)
    for ((i,j),b) in np.ndenumerate(bins):
        bins[i,j] = np.nonzero((x==i) & (y == j))[0]
    for ((i,j),b) in tqdm(np.ndenumerate(bins), total=bins.size):
        for (k,l), o in np.ndenumerate(bins):
            if abs(i-k) < 2 and abs(j-l) < 2 and len(o) and len(b):
                # Split B in bins based on the max product size
                inner_pairs = len(o) * len(b)
                batches = (inner_pairs - 1) // max_batch_size + 1
                for b_batch in np.array_split(b, batches):
                    left = pos[b_batch]
                    right = pos[o]
                    dist = cdist(pos[b_batch], pos[o])
                    k = distance_kernel(dist, radius)
                    left_idx, right_idx = np.nonzero(
                        np.all(
                            [
                            np.random.uniform(size=dist.shape) < k * p,
                            dist < radius,
                            b_batch[:,None] < o
                            ]
                            , axis = 0
                        )
                    ) 
                    if len(left_idx) > 0:
                        pairs += [
                            np.array([
                                b_batch[left_idx], 
                                o[right_idx]
                            ]).T
                        ]
    return np.concatenate(pairs)

def distance_kernel(x, r, p = 3.8):
    """
    Kernel with a smooth cutoff near r / 2.

    Keyword arguments:
    x   -- Argument where the kernel is evaluated
    r   -- Size of the kernel
    p   -- Controls smoothness of the cutoff, higher is sharper. 

    Returns:
    The value of the kernel at x. 
    """
    return 1. /  (1. + (2 * x / r) ** 3.8)

def nearest_distance(centers, pos):
    """
    Calculate the 
    """