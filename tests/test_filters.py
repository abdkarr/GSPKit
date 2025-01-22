from gspkit import data
from gspkit import filters

def test_gaussian_filter():
    graph = data.gen_graph(100, "er", {"p": 0.1})
    fltr_mat = filters.gaussian(graph)

    # Assert returned shapes are correct
    assert fltr_mat.shape == (100, 100), "Wrong shaped filter matrix is returned."

def test_heat_filter():
    graph = data.gen_graph(100, "er", {"p": 0.1})
    fltr_mat = filters.heat(graph)

    # Assert returned shapes are correct
    assert fltr_mat.shape == (100, 100), "Wrong shaped filter matrix is returned."

def test_tikhonov_filter():
    graph = data.gen_graph(100, "er", {"p": 0.1})
    fltr_mat = filters.tikhonov(graph)

    # Assert returned shapes are correct
    assert fltr_mat.shape == (100, 100), "Wrong shaped filter matrix is returned."
