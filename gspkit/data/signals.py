import networkx as nx
import numpy as np
import numpy.typing as npt

from numpy import linalg

from gspkit import typing
from gspkit import input_checks
from gspkit.linalg import laplacians

def gen_signals_from_signed_graph(
        G: nx.Graph, n_signals: int, fltr: str="gaussian", filter_params: dict = {},
        noise: float=0.1, rng: typing.RNG_TYPE = None
    ) -> npt.NDArray:
    """Generate graph signals from a signed graph.

    Graph signals are generated using the idea that signals are smooth with
    respect to positive edges and non-smooth with respect to negative edges as
    described in [1].

    Parameters
    ----------
    G : nx.Graph
        _description_
    n_signals : int
        _description_
    fltr : str, optional
        _description_, by default "gaussian"
    filter_params : dict, optional
        _description_, by default {}
    noise : float, optional
        _description_, by default 0.1
    rng : typing.RNG_TYPE, optional
        _description_, by default None

    Returns
    -------
    npt.NDArray
        _description_

    References
    ----------
    .. [1] Karaaslanli, Abdullah, and Selin Aviyente. "Dynamic Signed Graph
        Learning." ICASSP 2023-2023 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2023.
    """

    rng = input_checks.check_rng(rng)
    n_nodes = G.number_of_nodes()

    Lp = laplacians.get_pos_laplacian(G).toarray()
    Ln = laplacians.get_neg_laplacian(G).toarray()
    
    # Get the graph Laplacian spectrum
    ep, Vp = linalg.eigh(Lp)
    ep[ep < 1e-8] = 0

    en, Vn = linalg.eigh(Ln)
    en[en < 1e-8] = 0

    # Filters
    if fltr == "gaussian":
        hp = np.zeros(n_nodes)
        hp[ep > 0] = 1/np.sqrt(ep[ep>0])
        hn = np.zeros(n_nodes)
        hn[en > 0] = np.sqrt(en[en>0])
    elif fltr == "tikhonov":
        if "alpha" not in filter_params:
            filter_params["alpha"] = 10

        hp = 1/(1+filter_params["alpha"]*ep)
        hn = (1+filter_params["alpha"]*en)
    elif fltr == "heat":
        if "t" not in filter_params:
            filter_params["t"] = 10

        hp = np.exp(-filter_params["t"]*ep)
        hn = np.exp(filter_params["t"]*en)

    mid = n_nodes//2
    hp[mid:] = 0
    hn[:mid] = 0

    hp /= np.linalg.norm(hp)
    hn /= np.linalg.norm(hn)

    # Generate white noise
    X0 = rng.multivariate_normal(np.zeros(n_nodes), np.eye(n_nodes), n_signals).T
    X0p = np.diag(hp)@Vp.T@X0
    X0n = np.diag(hn)@Vn.T@X0
    
    X = 0.5*(Vp@X0p + Vn@X0n)

    # Add noise
    X_norm = np.linalg.norm(X)
    E = rng.normal(0, 1, X.shape)
    E_norm = np.linalg.norm(E)
    X += E*(noise*X_norm/E_norm)

    return X