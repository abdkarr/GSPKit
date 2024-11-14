import networkx as nx
import numpy as np

from scipy.spatial.distance import pdist, squareform

from gspkit import typing
from gspkit import input_checks
from gspkit import exceptions

def ensure_connectedness(generator):
    """Draw a random graph until finding a connected one.
    """

    i = 1
    while True:
        G = generator()
        if nx.is_connected(G):
            return G
        else:
            i += 1

        if i>500:
            return None

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

def gen_graph(
        n_nodes: int, model: str, model_params: dict, ensure_connected: bool = True, 
        rng: typing.RNG_TYPE = None
    ):
    """Generate a random graph from a random graph model.

    This function is implemented to provide a single API point that can be
    called to generate graphs from different random graph models. 

    Parameters
    ----------
    n_nodes : int
        Number of nodes. 
    model : str
        The random graph model to use for generating the graph. Currently, the
        followings are supported:

            - `er`: Erdős–Rényi graph. See `nx.erdos_renyi_graph` for the
              details.
            - `ba`: Barabási–Albert graph. See `nx.barabasi_albert_graph` for
              the details.
            - `rgg`: Random geometric graph. See `gen_random_geometric_graph`
              for the details.

    model_params : dict
        The dictionary of parameters for the random graph model. Parameters of
        the models differ and they are listed below:

            - `er`: `model_params` should have element `p` representing edge
              probility. 
            - `ba`: `model_params` should have element `m` representing growth
              parameter.
            - `rgg`: `model_params` should have elements `sigma` and `th`
              representing scale of RBF kernel and threshold. 

    ensure_connected : bool, optional
        Flag to ensure the generated graph is connected. The function keeps 
        generating a new graph until generating a connected one. The function 
        stops searching a connected graph generating 500 graphs and returns an 
        error. By default True.
    rng : typing.RNG_TYPE, optional
        Random number generator. If one wants the function to return the same 
        output every time, this needs to be set. By default None.

    Returns
    -------
    nx.Graph
        Generated graph.

    Raises
    ------
    exceptions.MaxIterReachedException
        When connected cannot be ensured.
    """
    
    rng = input_checks._check_rng(rng)

    if model == "er":
        generator = lambda: nx.erdos_renyi_graph(n_nodes, model_params["p"], seed=rng)
    elif model == "ba":
        generator = lambda: nx.barabasi_albert_graph(n_nodes, model_params["m"], seed=rng) 
    elif model == "rgg":
        generator = lambda: gen_random_geometric_graph(
            n_nodes, model_params["sigma"], model_params["th"], seed=rng
        )

    if ensure_connected:
        G = ensure_connectedness(generator)
        if G is None:
            raise exceptions.MaxIterReachedException(
                "I cannot create a connected graph with the given model parameters."
            )

    return G