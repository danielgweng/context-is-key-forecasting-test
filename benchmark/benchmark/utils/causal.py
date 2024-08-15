import networkx as nx
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import scipy


def check_dagness(inst_graph):
    """
    Check for acylcity of the instantaneous graph
    """
    num_nodes = inst_graph.shape[0]
    error_message = colored(
        "Error in DAG generation: Instantaneous graph is not acyclic!", "red"
    )
    dag_constraint = np.trace(scipy.linalg.expm(inst_graph * inst_graph)) - num_nodes
    assert dag_constraint == 0, error_message


def plot_temporal_graph(complete_graph):
    """
    Function to visualize the instantaneous and lagged graphs, to aid in debugging
    """
    L = complete_graph.shape[0] - 1
    d = complete_graph.shape[1]
    G = nx.DiGraph()

    for lag in range(L + 1):
        for i in range(d):
            G.add_node((lag, i), label=f"{i}")

    # Add edges based on the adjacency matrices
    for lag in range(L + 1):
        for i in range(d):
            for j in range(d):
                if complete_graph[lag, i, j] == 1:
                    G.add_edge(
                        (L - lag, i), (L, j)
                    )  # Connection from (t-lag, variable i) to (t, variable j)

    pos = {}
    for lag in range(L + 1):
        for i in range(d):
            pos[(lag, i)] = (lag, -i)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={(lag, i): f"{i}" for lag in range(L + 1) for i in range(d)},
        node_size=500,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        ax=ax,
    )

    # Set labels for each column
    timestep_labels = [f"T-{L-lag}" if lag != L else "T" for lag in range(L + 1)]
    for lag, label in enumerate(timestep_labels):
        ax.text(
            lag,
            -d + 0.3,
            label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="bold",
            color="green",
        )

    ax.set_title("Temporal Adjacency Matrix Visualization")
    plt.show()
    plt.savefig("temporal_graph.png")
