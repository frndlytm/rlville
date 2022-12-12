import numpy as np
from matplotlib import pyplot


def plot_curves(
    title: str,
    xlabel: str,
    ylabel: str,
    curves: list[np.array],
    legend: list[str],
    colors: list[str],
):
    """
    :param title: Title for the plot
    :type title: str

    :param xlabel: Label for the x-axis
    :type xlabel: str

    :param ylabel: Label for the y-axis
    :type ylabel: str

    :param curves: results arrays to plot
    :type curves: list[np.array]

    :param legend: labels corresponding to each result array
    :type legend: list[str]

    :param colors: labels corresponding to each result array
    :type colors: list[str]
    """
    # set the figure type
    n_curves = len(curves)
    assert len(legend) == n_curves and len(colors) == n_curves
    fig, ax = pyplot.subplots(figsize=(12, 8))
    handles = []

    for i in range(n_curves):
        curve, label, color = curves[i], legend[i], colors[i]

        # compute the standard error
        trials, steps = curve.shape
        xs = list(range(steps))
        ys = curve.mean(axis=0)

        # plot the confidence band
        error = (curve.std(axis=0) / np.sqrt(trials)) * 1.96
        h, = ax.plot(xs, curve.mean(axis=0), label=label, color=color)
        ax.fill_between(xs, ys - error, ys + error, alpha=0.3, color=color)

        handles.append(h)

    # plot legends
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=handles)

    fig.savefig(f"{title}.png")
    pyplot.show()
