from gspkit.data import gen_graph
from gspkit.data import gen_signals_from_signed_graph

def test_gen_smooth_signals_function():
    graph = gen_graph(100, "signed-er")
    signals = gen_signals_from_signed_graph(graph, 1000)

test_gen_smooth_signals_function()