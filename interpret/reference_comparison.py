"""Model predictions against the measurements they are supposed to reproduce.

The question this answers — does the model follow a trend somebody already measured — is not specific to any property
or any material. A panel is a shared x axis, one predicted series per model, and a set of reference points that may
sit at their own x values, because a measurement rarely lands exactly on the composition that was modelled.

Nothing is read from a file and nothing is computed here: the caller brings numbers. What belongs to a domain is the
series it chooses to plot, which is why the panels are an argument.
"""

import math
from typing import Mapping, NamedTuple, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


class ComparisonPanel(NamedTuple):
    """One panel of a predictions-against-measurements figure.

    Attributes:
        title: Panel heading, e.g. the name of the series being swept.
        x_label: What the shared x axis measures, units included.
        x: The x coordinate of every predicted point.
        predictions: {series name: predicted values, aligned with `x`}. The names become the legend.
        measured_x: x coordinates of the reference points, which need not match `x`.
        measured_y: The measured values themselves.
    """

    title: str
    x_label: str
    x: Sequence[float]
    predictions: Mapping[str, Sequence[float]]
    measured_x: Sequence[float]
    measured_y: Sequence[float]


def plot_predictions_against_measurements(
    panels: Sequence[ComparisonPanel],
    *,
    y_label: str,
    measurement_label: str = "literature",
    max_columns: int = 3,
    panel_size: Tuple[float, float] = (6.7, 4.0),
    save_path=None,
) -> None:
    """Plot every panel side by side, predictions and measurements overlaid.

    The grid grows with the number of panels rather than the figure being stretched: panels wrap onto a new row past
    `max_columns`, so twelve series stay readable where a single row of twelve would be eighty inches wide. Unused
    cells in the last row are removed rather than left as empty axes.

    Args:
        panels: The panels to draw, in order.
        y_label: Shared y-axis label; every panel plots the same quantity.
        measurement_label: Legend entry for the reference points.
        max_columns: Most panels in one row.
        panel_size: (width, height) in inches for a single panel.
        save_path: If set, save the figure here instead of showing it.

    Raises:
        ValueError: If no panel is given, or a panel carries no predicted series, since either would draw a figure
            that says nothing.
    """
    if not panels:
        raise ValueError("plot_predictions_against_measurements needs at least one panel.")

    empty = [panel.title for panel in panels if not panel.predictions]
    if empty:
        raise ValueError(f"No predicted series for panel(s): {empty}.")

    columns = min(len(panels), max(1, max_columns))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(panel_size[0] * columns, panel_size[1] * rows), squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    for ax, panel in zip(flat, panels):
        _plot_one_panel(ax, panel, y_label=y_label, measurement_label=measurement_label)
    for ax in flat[len(panels):]:
        fig.delaxes(ax)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _plot_one_panel(ax, panel: ComparisonPanel, *, y_label: str, measurement_label: str) -> None:
    """Draw one panel's predicted series and its reference points onto `ax`.

    The legend is built from the panel's own series names, so it stays correct whichever models were compared.
    """
    for values in panel.predictions.values():
        sns.scatterplot(x=panel.x, y=values, ax=ax)
    sns.scatterplot(x=panel.measured_x, y=panel.measured_y, ax=ax)

    ax.set_title(panel.title, fontsize=16)
    ax.set_xlabel(panel.x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    legend = ax.legend([*panel.predictions, measurement_label], loc="upper right", fontsize=12)
    legend.get_frame().set_facecolor("white")


def panel_from_series(
    title: str,
    x_label: str,
    x: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    measurements: Mapping[float, float],
) -> ComparisonPanel:
    """Build a panel where the measurements arrive as {x: measured value}.

    A convenience for the common case of a reference table keyed by the same quantity the x axis shows.
    """
    return ComparisonPanel(
        title=title,
        x_label=x_label,
        x=x,
        predictions=predictions,
        measured_x=list(measurements.keys()),
        measured_y=list(measurements.values()),
    )
