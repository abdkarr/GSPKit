from typing import Callable

import networkx as nx
import numpy as np

from scipy.spatial.distance import pdist, squareform

from gspkit import typing
from gspkit import input_checks
from gspkit import exceptions


def ensure_connectedness(generator: Callable) -> None | nx.Graph:
    r"""Draw a random graph until finding a connected one."""

    # TODO Testing

    i = 1
    while True:
        G = generator()
        if nx.is_connected(G):
            return G
        else:
            i += 1

        if i > 500:
            return None


def assign_signs(G: nx.Graph, fraction: float, rng: typing.RNGType = None):
    r"""Convert an unsigned graph to signed one by random sign assignment.

    Given an unsigned graph G, this function generates a signed graph by
    assigning a negative sign to a randomly selected fraction of G's edges and
    keeping the other edges as positive.

    Parameters
    ----------
    G : nx.Graph
        Input graph. Modifies G in-place.
    fraction : float
        Fraction of edges to be set as negative edge.
    rng : typing.RNGType, optional
        Random number generator. If one wants the function to return the same
        output every time, this needs to be set. By default None.
    """

    # TODO Testing

    rng = input_checks.check_rng(rng)

    nx.set_edge_attributes(
        G, {e: -1 if rng.binomial(1, fraction) else 1 for e in G.edges}, "sign"
    )


def gen_er(n_nodes: int, p: float, rng: typing.RNGType = None) -> nx.Graph:
    """Generate a graph from Erdős-Rényi random graph model.

    This function is just a wrapper around NetworkX
    [erdos_renyi_graph][networkx.generators.random_graphs.erdos_renyi_graph]
    implemented to make its API consistent with the other graph generation
    functions.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    p : float
        Edge inclusion probability.
    rng : typing.RNGType, optional
        Random number generator.

    Returns
    -------
    graph :
        Generated graph.
    """

    rng = input_checks.check_rng(rng)
    return nx.erdos_renyi_graph(n_nodes, p, seed=rng)


def gen_ba(n_nodes: int, m: int, rng: typing.RNGType = None) -> nx.Graph:
    """Generate a graph from Barabási–Albert random graph model.

    This function is just a wrapper around NetworkX
    [barabasi_albert_graph][networkx.generators.random_graphs.barabasi_albert_graph]
    implemented to make its API consistent with the other graph generation
    functions.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    m : int
        Number of edges to attach to newly added node.
    rng : typing.RNGType, optional
        Random number generator.

    Returns
    -------
    graph :
        Generated graph.
    """

    rng = input_checks.check_rng(rng)
    return nx.barabasi_albert_graph(n_nodes, m, seed=rng)


def gen_signed_er(
    n_nodes: int, p: float, frac: float, rng: typing.RNGType = None
) -> nx.Graph:
    """Generate a graph from signed Erdős-Rényi random graph model.

    This function first generates a graph from Erdős-Rényi model. Then, it
    sets `frac` fraction of the generated graph edges as negative edges, while
    the other edges are considered as positive edges.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    p :
        Edge inclusion probability of Erdős-Rényi model.
    frac :
        Fraction of edges to be set as negative edge.
    rng :
        Random number generator.

    Returns
    -------
    graph :
        Generated graph. Its edges has attribute `sign`, which is set to `1` for
        positive edges, and `-1` for negative edges.
    """
    rng = input_checks.check_rng(rng)

    graph = gen_er(n_nodes, p, rng)
    assign_signs(graph, frac, rng)

    return graph


def gen_signed_ba(
    n_nodes: int, m: int, frac: float, rng: typing.RNGType = None
) -> nx.Graph:
    """Generate a graph from signed Barabási–Albert random graph model.

    This function first generates a graph from Barabási–Albert model. Then, it
    sets `frac` fraction of the generated graph edges as negative edges, while
    the other edges are considered as positive edges.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    m : int
        Number of edges to attach to newly added node.
    frac :
        Fraction of edges to be set as negative edge.
    rng :
        Random number generator.

    Returns
    -------
    graph :
        Generated graph. Its edges has attribute `sign`, which is set to `1` for
        positive edges, and `-1` for negative edges.
    """
    rng = input_checks.check_rng(rng)

    graph = gen_ba(n_nodes, m, rng)
    assign_signs(graph, frac, rng)

    return graph


def gen_rgg(
    n_nodes: int,
    sigma: float,
    th: float,
    rng: typing.RNGType = None,
) -> nx.Graph:
    r"""Generate a random geometric graph.

    A random geometric graph is a graph generated from a 2-dimensional point
    cloud where points are drawn uniformly from unit square $[0, 1]^2$.
    In the generated graph, each point is a node and two nodes are connected
    with a binary edge if their similarity is larger than a threshold.
    Similarity of points are measured using RBF kernel. The implementation is
    based on [Kalofolias16].

    Parameters
    ----------
    n_nodes : 
        Number of nodes.
    sigma :
        Scale of the RBF kernel.
    th :
        Threshold used to determine which nodes are connected.
    rng :
        Random number generator. 

    Returns
    -------
    graph :
        Generated graph. 
    """

    points = rng.uniform(0, 1, size=(n_nodes, 2))

    dists = pdist(points, "sqeuclidean")
    rbf = np.exp(-dists / (sigma**2))

    rbf[rbf < th] = 0

    return nx.from_numpy_array(squareform(rbf))


def gen_graph(
    n_nodes: int,
    model: typing.GraphModelType | str,
    model_params: dict,
    ensure_connected: bool = True,
    rng: typing.RNGType = None,
) -> nx.Graph:
    r"""Generate a graph from a given graph model.

    This function is implemented to provide a single API point that can be
    called to generate graphs from different graph models.

    Parameters
    ----------
    n_nodes :
        Number of nodes.
    model :
        The graph model to use for generating the graph. Available options are
        listed below and their details can be found at
        [GraphModelType][gspkit.typing.GraphModelType].

        ??? note "Available Options"
            ```python exec="1" from gspkit import typing for t in
            typing.GraphModelType:
                print(f"- `#!python '{t.value}'`")
            ```

    model_params :
        The dictionary of parameters for the graph model. Parameters of the
        models differ and see model details for the parameters accepted by
        different models.

        !!! note
            Model implementations accept `rng` to set seed of random number
            generator. If `model_params` includes `rng`, it will be ignored.
            Seed setting should be done through `rng` argument of this function.

    ensure_connected :
        Flag to ensure the generated graph is connected. The function keeps
        generating a new graph until generating a connected one. The function
        stops searching a connected graph after generating 500 graphs and
        returns an error. For signed graphs, connectedness is defined in terms
        of topology (signed of edges are ignored). By default True.
    rng : typing.RNGType, optional
        Random number generator.

    Returns
    -------
    graph :
        Generated graph. The returned graph might have node/edge attributed
        depending on `model`. Check the details of model used for these
        attributes.
    """

    # Check if graph model is valid
    if isinstance(model, str):
        try:
            model = typing.GraphModelType(model)
        except ValueError:
            raise ValueError("Invalid graph model.")
    model = model.value

    rng = input_checks.check_rng(rng)
    model_params["rng"] = rng

    model_mapping = {
        "er": lambda: gen_er(n_nodes, **model_params),
        "ba": lambda: gen_ba(n_nodes, **model_params),
        "rgg": lambda: gen_rgg(n_nodes, **model_params),
        "signed-er": lambda: gen_signed_er(n_nodes, **model_params),
        "signed-ba": lambda: gen_signed_ba(n_nodes, **model_params),
    }

    generator = model_mapping[model]

    if ensure_connected:
        graph = ensure_connectedness(generator)
        if graph is None:
            raise exceptions.MaxIterReachedException(
                "I cannot create a connected graph with the given model parameters."
            )
    else: 
        graph = generator()

    return graph
