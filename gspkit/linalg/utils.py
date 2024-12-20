import numpy as np
import networkx as nx
import numpy.typing as npt

from scipy.sparse import csr_array

def rowsum_mat(n):
    r"""Construct row-sum matrix for a symmetric zero-diagonal matrix.

    For a symmetric matrix :math:`\mathbf{A} \in \mathbb{R}^{n \\times n}`, let
    :math:`\mathbf{a} \in \mathbb{R}^{n(n-1)/2}` be its strictly upper
    triangular part in vector form. Row sum matrix :math:`\mathbf{S} \in
    \mathbb{R}^{n \\times n(n-1)/2}` can be used as :math:`\mathbf{S}\mathbf{a}`
    to calculate :math:`\mathbf{A}\mathbf{1} - \mathrm{diag}(\mathbf{A})`, where
    :math:`\mathbf{1}` is :math:`n` dimensional all-ones vector and
    :math:`\mathrm{diag}` is the operator returning diagonal of the input matrix
    as a vector.
    
    Parameters
    ----------
    n : int
        Dimension of the matrix.

    Returns
    -------
    scipy.sparse.csr_array
        Matrix to be used in row-sum calculation.
    """

    # TODO Testing

    i, j = np.triu_indices(n, k=1)
    M = len(i)
    rows = np.concatenate((i, j))
    cols = np.concatenate((np.arange(M), np.arange(M)))

    return csr_array((np.ones((2*M, )), (rows, cols)), shape=(n, M))

def vectorize_a_graph(
        G: nx.Graph, signed: bool=False
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    r"""Get strictly upper triangular part of the graph adjacency as a vector. 

    The function can handle unsigned and signed graphs. In the latter case it
    returns two vectors representing strictly upper triangular part of positive
    and negative adjacency matrix. 

    Parameters
    ----------
    G : nx.Graph
        Input graph. 
    signed : bool, optional
        Flag indicating whether input graph is signed or not. If True, `G` should
        have edge attribute named `signed` indicating edges' sign.

    Returns
    -------
    np.NDArray or tuple of np.NDArrays
        Output vector or vectors in case of signed graph
    """

    # TODO Testing

    n_nodes = G.number_of_nodes()

    if signed:
        w = nx.to_numpy_array(G, weight="sign")[np.triu_indices(n_nodes, k=1)]
        w_pos = w.copy()
        w_pos[w_pos < 0] = 0
        w_neg = w.copy()
        w_neg[w_neg > 0] = 0
        return w_pos, np.abs(w_neg)
    else:
        return nx.to_numpy_array(G)[np.triu_indices(n_nodes, k=1)]