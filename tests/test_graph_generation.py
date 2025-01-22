from gspkit.data import gen_graph

def test_gen_graph_function():
    er_params = {"p": 0.1}
    G = gen_graph(100, "er", er_params)

    ba_params = {"m": 5}
    G = gen_graph(100, "ba", ba_params)

    rgg_params = {"sigma": 0.25, "th": 0.6}
    G = gen_graph(100, "rgg", rgg_params)

    signed_er_params = {"p": 0.1, "frac": 0.5}
    G = gen_graph(100, "signed-er", signed_er_params)

    signed_ba_params = {"m": 5, "frac": 0.5}
    G = gen_graph(100, "signed-ba", signed_ba_params)