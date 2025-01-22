import numpy as np

from gspkit.linalg import spectrum
from gspkit import data

def test_get_laplacian_spectrum():
    graph = data.gen_graph(100, "er", {"p": 0.1})
    eig_vals, eig_vecs = spectrum.get_laplacian_spectrum(graph)

    # Assert return shapes are correct
    assert eig_vals.shape == (100, ), "Wrong shaped array for eigenvalues."
    assert eig_vecs.shape == (100, 100), "Wrong shaped array for eigenvalues."

    # Assert minimum eigenvalue is zero
    assert np.min(eig_vals) == 0, "Smallest eigenvalue isn't ensured to be 0."