# Filters

::: gspkit.filters.gaussian

<!-- ??? Example

    ```python exec="1" source="tabbed-right" html="1"
    from io import StringIO

    import seaborn as sns
    import matplotlib.pyplot as plt

    from numpy.linalg import eigh

    from gspkit.data import gen_graph
    from gspkit import filters
    from gspkit import STYLE_FILE

    sns.set_theme(context="paper", style="whitegrid", palette="Set2")
    plt.style.use(STYLE_FILE)

    graph = gen_graph(100, "er")
    fltr_mat = filters.gaussian(graph)

    eig_vals, eig_vecs = eigh(fltr_mat)

    plt.plot(eig_vals, ".")

    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    print(buffer.getvalue())
    ``` -->

::: gspkit.filters.heat

::: gspkit.filters.tikhonov