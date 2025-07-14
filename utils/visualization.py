import matplotlib.pyplot as plt


def scatter(df, x_col, y_col, s=8, color_col=None, title=None):
    """
    Create a scatter plot from a DataFrame.
    Returns a matplotlib figure.

    Parameters:
    - df: DataFrame containing the data.
    - x_col: Column name for x-axis.
    - y_col: Column name for y-axis.
    - color_col: Optional column name for coloring points.
    - title: Optional title for the plot.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col] if color_col else None,
        cmap="inferno",
        alpha=0.6,
        edgecolors="w",
        s=s,
    )

    if color_col:
        plt.colorbar(scatter, ax=ax, label=color_col)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)

    return fig
