from gspkit.data import gen_graph

def test_gen_graph_function():
    G = gen_graph(100, "er")

    G = gen_graph(100, "ba")

    G = gen_graph(100, "rgg")

    G = gen_graph(100, "signed-er")

    G = gen_graph(100, "signed-ba")