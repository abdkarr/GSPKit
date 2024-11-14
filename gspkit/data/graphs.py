import networkx as nx
import numpy as np

from scipy.spatial.distance import pdist, squareform

from gspkit import typing
from gspkit import input_checks

def gen_random_geometric_graph(
        n_nodes: int, sigma: float=0.25, th: float=0.6, rng: typing.RNG_TYPE=None
    ):
    """Generate a random geometric graph. 

    A random geometric graph is a graph generated from a 2D point clouds where
    points are drawn uniformly from unit square :math:`[0, 1]x[0, 1]`. In the
    generated graph, each point is a node and two nodes are connected with a
    binary edge if their similarity is larger than a threshold. Similarity of
    points are measured using RBF kernel.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    sigma : float, optional
        Scale of the RBF kernel, by default 0.25.
    th : float, optional
        Threshold used to determine which nodes are connected, by default 0.6.
    rng : typing.RNG_TYPE
        Random number generator. If one wants the function to return the same 
        output every time, this needs to be set. By default None.

    Returns
    -------
    nx.Graph
        Generated graph.
    """

    points = rng.uniform(0, 1, size=(n_nodes, 2))

    dists = pdist(points, "sqeuclidean")
    rbf = np.exp(-dists/(sigma**2))
    
    rbf[rbf < th] = 0
    
    return nx.from_numpy_array(squareform(rbf))