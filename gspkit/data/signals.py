import networkx as nx
import numpy as np
import numpy.typing as npt

from numpy import linalg

from gspkit import typing
from gspkit import input_checks
from gspkit.linalg import laplacians

def gen_signals_from_signed_graph(
        G: nx.Graph, n_signals: int, fltr: str="gaussian", fltr_params: dict = {},
        noise: float=0.1, rng: typing.RNG_TYPE = None
    ) -> npt.NDArray:
    r"""Generate graph signals from a signed graph.

    Graph signals are generated using the idea that signals are smooth with
    respect to positive edges and non-smooth with respect to negative edges as
    described in [1]. Namely, let :math:`\mathbf{L}_{\rm p}` and
    :math:`\mathbf{L}_{\rm n}` be the positive and negative Laplacians of the
    signed graph. A graph signal is generated as follows:
    
    .. math::

        \mathbf{x} = 
            \sum_{i=1}^{\lfloor N/2 \rfloor} 
                \mathbf{v}_i^{\rm p} h(\lambda_i^{\rm p}) {\mathbf{v}_i^{\rm
                p}}^\top \mathbf{x}_0 +
            \sum_{i=\lceil N/2 \rceil}^{N} 
                \mathbf{v}_i^{\rm n} h(\lambda_i^{\rm n}) {\mathbf{v}_i^{\rm
                n}}^\top \mathbf{x}_0,

    where :math:`(\lambda_i^{\rm p}, \mathbf{v}_i^{\rm p})` and
    :math:`(\lambda_i^{\rm n}, \mathbf{v}_i^{\rm n})` are :math:`i`th eigenpair
    of :math:`\mathbf{L}_{\rm p}` and :math:`\mathbf{L}_{\rm n}`, respectively.
    :math:`\mathbf{x}_0` is the input signal drawn :math:`N`-dimensional
    standard normal distribution. :math:`h(\lambda_i^{\rm n})` is the filters as
    detailed below.

    Graph filters used in this function are developed for smooth graph signal
    generation. These filters are modified to generate non-smooth signals by
    changing the spectrum of the filters to be high-pass. These modifications
    are as follows:

    - **Gaussian**: Smooth version is
      :math:`h(\lambda_i)=\frac{1}{\sqrt{\lambda_i}}`. Non-smooth version is
      then :math:`h(\lambda_i)=\sqrt{\lambda_i}`.
    - **Heat**: Smooth version is :math:`h(\lambda_i)=\exp(-t\lambda_i)` where
      :math:`t` is the filter parameter. Non-smooth version is then
      :math:`h(\lambda_i)=exp(t\lambda_i)`.
    - **Tikhonov**: Smooth version is
      :math:`h(\lambda_i)=\frac{1}{1+\alpha\lambda_i}}` where :math:`\alpha` is
      the filter parameter. Non-smooth version is then
      :math:`h(\lambda_i)=1+\alpha\lambda_i`.

    Modification to generalize these filters to mon-smooth versions are ad-hoc
    and there is a need for their rigorous investigation.

    Parameters
    ----------
    G : nx.Graph
        Input signed graph. Its edges must have an attributed named `sign` set
        to 1 or -1 indicating sign of the edge.
    n_signals : int
        Number of signals to generate.
    fltr : str, optional
        Graph filter to use, by default "gaussian". The available options are as
        follows:

        - `gaussian`: Generate the signals using a Gaussian graph filter.
        - `heat`: Generate the signals using a Heat graph filter.
        - `tikhonov`: Generate the signals using a Tikhonov graph filter.

    filter_params : dict, optional
        Parameters of the graph filters. Available parameters are:

        - `alpha`: Parameter of Tikhonov filter, by defualt 10.
        - `t`: Parameter of Heat filter, by default 10.

    noise : float, optional
        Amount of Gaussian noise to add to generated graph signals, by default
        0.1. Amount of noise is determined in L2-sense, that is L2 norm of the
        noise is `noise` fraction of the L2 norm of the clean signals.
    rng : typing.RNG_TYPE, optional
        Random number generater, by default None.

    Returns
    -------
    npt.NDArray
        Generated signals as a matrix whose columns are individual graph
        signals.

    References
    ----------
    .. [1] Karaaslanli, Abdullah, and Selin Aviyente. "Dynamic Signed Graph
        Learning." ICASSP 2023-2023 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2023.
    """

    # TODO Testing

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
        if "alpha" not in fltr_params:
            fltr_params["alpha"] = 10

        hp = 1/(1+fltr_params["alpha"]*ep)
        hn = (1+fltr_params["alpha"]*en)
    elif fltr == "heat":
        if "t" not in fltr_params:
            fltr_params["t"] = 10

        hp = np.exp(-fltr_params["t"]*ep)
        hn = np.exp(fltr_params["t"]*en)

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