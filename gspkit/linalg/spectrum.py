import networkx as nx
import numpy.typing as npt

from numpy.linalg import eigh


def get_laplacian_spectrum(graph: nx.Graph) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Get Laplacian spectrum of the input graph.

    !!! warning
        This function employs dense version of eigendecomposition. So it is not
        scalable to large graphs.

    Parameters
    ----------
    graph :
        Input graph.

    Returns
    -------
    eig_vals : npt.NDArray
        (N, ) dimensional array of the graph Laplacian eigenvalues in ascending
        order.
    eig_vecs : npt.NDArray
        (N, N) dimensional matrix, whose columns are the graph Laplacian
        eigenvectors. `#!python eig_vecs[:, i]` corresponds to `#!python
        eig_vals[i]`.
    """

    laplacian = nx.laplacian_matrix(graph).todense()
    eig_vals, eig_vecs = eigh(laplacian)

    # Ensure small eigenvalues are zero as they should be
    eig_vals[eig_vals < 1e-8] = 0

    return eig_vals, eig_vecs