import numpy as np
import numpy.typing as npt
import networkx as nx

from gspkit.linalg import get_laplacian_spectrum


def gaussian(graph: nx.Graph, shift: float = 0) -> npt.NDArray:
    r"""Construct a Gaussian filter defined on the input graph.

    ??? Definition
        Gaussian filter definition is based on [Kalofolias16]. Namely, let
        $\mathbf{L}$ be the graph Laplacian:

        $$ h(\mathbf{L}) = \left(
            \frac{1}{\lambda_{max}}\mathbf{L} + s\mathbf{I}
        \right)^{-1/2}, 
        $$

        where $s$ is the filter parameter.

    Parameters
    ----------
    graph :
        Input graph.
    shift :
        Filter parameter.

    Returns
    -------
    filter_mat :
        (N, N) filter matrix $h(\mathbf{L})$, where N is the number of nodes.
    """

    eig_vals, eig_vecs = get_laplacian_spectrum(graph)

    # Eigenvector corresponding to 0 is constant, so it is DC in signal processing sense
    dc_component = np.where(eig_vals == 0)[0]

    # Shift and normalize eigenvalues to [0, 2]
    e_max = np.max(eig_vals)
    eig_vals /= e_max

    # Construct the filter in Fourier domain
    h = np.zeros_like(eig_vals)
    h[eig_vals > 0] = 1 / np.sqrt(eig_vals[eig_vals > 0] + shift)

    # Remove the DC component from spectrum
    h[dc_component] = 0

    # Construct the filter matrix in vertex domain
    fltr_mat = eig_vecs @ np.diag(h) @ eig_vecs.T

    return fltr_mat


def heat(graph: nx.Graph, alpha: int = 5) -> npt.NDArray:
    r"""Construct a Heat kernel filter defined on the input graph.

    ??? Definition
        Heat kernel definition is based on [Kalofolias16], which defines it as
        follows:

        $$ h(\mathbf{L}) = \exp\left(
            - \frac{\alpha}{\lambda_{max}}\mathbf{L} 
        \right), 
        $$

        where $\mathbf{L}$ is the graph Laplacian, and $\alpha$ is the filter
        parameter.

    Parameters
    ----------
    graph :
        Input graph.
    alpha :
        Filter parameter.

    Returns
    -------
    filter_mat :
        (N, N) filter matrix $h(\mathbf{L})$, where N is the number of nodes.
    """

    eig_vals, eig_vecs = get_laplacian_spectrum(graph)

    # Eigenvector corresponding to 0 is constant, so it is DC in signal processing sense
    dc_component = np.where(eig_vals == 0)[0]

    # Shift and normalize eigenvalues to [0, 2]
    e_max = np.max(eig_vals)
    eig_vals /= e_max

    # Construct the filter in Fourier domain
    h = np.exp(-alpha * eig_vals)

    # Remove the DC component from spectrum
    h[dc_component] = 0

    # Construct the filter matrix in vertex domain
    fltr_mat = eig_vecs @ np.diag(h) @ eig_vecs.T

    return fltr_mat


def tikhonov(graph: nx.Graph, alpha: int = 5) -> npt.NDArray:
    r"""Construct a Tikhonov filter defined on the input graph.

    ??? Definition
        Tikhonov filter definition is based on [Kalofolias16], which defines it as
        follows:

        $$
            h(\mathbf{L}) = \left(
                \mathbf{I} + \frac{\alpha}{\lambda_{max}}\mathbf{L}
            \right)^{-1}
        $$

        where $\mathbf{L}$ is the graph Laplacian, and $\alpha$ is the filter
        parameter.

    Parameters
    ----------
    graph :
        Input graph.
    alpha :
        Filter parameter.

    Returns
    -------
    filter_mat : 
        (N, N) filter matrix $h(\mathbf{L})$, where N is the number of nodes.
    """

    eig_vals, eig_vecs = get_laplacian_spectrum(graph)

    # Eigenvector corresponding to 0 is constant, so it is DC in signal processing sense
    dc_component = np.where(eig_vals == 0)[0]

    # Shift and normalize eigenvalues to [0, 1]
    e_max = np.max(eig_vals)
    eig_vals /= e_max

    # Construct the filter in Fourier domain
    h = 1 / (1 + alpha * eig_vals)

    # Remove the DC component from spectrum
    h[dc_component] = 0

    # Construct the filter matrix in vertex domain
    fltr_mat = eig_vecs @ np.diag(h) @ eig_vecs.T

    return fltr_mat
