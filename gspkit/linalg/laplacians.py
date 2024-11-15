import numpy as np
import networkx as nx

from scipy import sparse

def get_pos_laplacian(G: nx.Graph) -> sparse.sparray:
    """Get the Laplacian of the positive part of a signed graph. 

    The function construct an unsigned graph from positive part of the input
    signed graph and returns its Laplacian matrix.

    Parameters
    ----------
    G : nx.Graph
        Input signed graph. Its edge must have attribute "sign" with values set
        to 1 or -1 indicating sign of the edge.

    Returns
    -------
    sparse.sparray
        Laplacian matrix. 
    """
    
    A = nx.adjacency_matrix(G, weight="sign")
    
    A[A<0] = 0
    L = sparse.diags_array(np.sum(A, axis=1)) - A

    return L

def get_neg_laplacian(G: nx.Graph) -> sparse.sparray:
    """Get the Laplacian of the negative part of a signed graph. 

    The function construct an unsigned graph from negative part of the input
    signed graph and returns its Laplacian matrix.

    Parameters
    ----------
    G : nx.Graph
        Input signed graph. Its edge must have attribute "sign" with values set
        to 1 or -1 indicating sign of the edge.

    Returns
    -------
    sparse.sparray
        Laplacian matrix. 
    """
    A = nx.adjacency_matrix(G, weight="sign")
    
    A[A>0] = 0
    L = sparse.diags_array(np.sum(A, axis=1)) - A

    return L