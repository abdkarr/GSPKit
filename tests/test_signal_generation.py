from gspkit.data import gen_graph, gen_signals

def test_gen_smooth_signals_function():
    graph = gen_graph(100, "er", {"p": 0.1})
    signals = gen_signals(graph, 1000)