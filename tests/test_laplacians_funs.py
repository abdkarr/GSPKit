import numpy as np

from gspkit.data import gen_graph
from gspkit.linalg import laplacians

def test_get_pos_neg_laplacian():
    graph = gen_graph(100, "signed-er")
    Lp = laplacians.get_pos_laplacian(graph)
    Ln = laplacians.get_neg_laplacian(graph)