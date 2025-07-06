# generate data (Y, X1, X2, X3, X4, D, Y(1), Y(0))
# E[Y(1)-Y(0)|X] > 0 iff X2 > X1

import numpy as np
import pandas as pd


def generate_x(n: int) -> pd.DataFrame:
    """
    Generate independent uniform [0, 1] data for X1, X2, X3, and X4.
    Each column represents a feature, and all features are independent.
    Args:
        n (int): Number of samples to generate.
    Returns:
        pd.DataFrame: DataFrame containing the generated features X1, X2, X3, and X4.
    """
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    x3 = np.random.uniform(0, 1, n)
    x4 = np.random.uniform(0, 1, n)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4})


def generate_x_on_grid(n: int, grid_size: int = 100) -> pd.DataFrame:
    x1 = np.linspace(0, 1, grid_size)
    x2 = np.linspace(0, 1, grid_size)
    x3 = np.linspace(0, 1, grid_size)
    x4 = np.linspace(0, 1, grid_size)
    x1, x2, x3, x4 = np.meshgrid(x1, x2, x3, x4)
    x1 = x1.flatten()[:n]
    x2 = x2.flatten()[:n]
    x3 = x3.flatten()[:n]
    x4 = x4.flatten()[:n]
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4})


def add_potential_outcomes(
    df: pd.DataFrame,
    error_variance: float = 1.0,
) -> pd.DataFrame:
    df["epsilon_0"] = np.random.normal(0, error_variance, len(df))
    df["epsilon_1"] = np.random.normal(0, error_variance, len(df))
    df["Y(0)"] = 0.7 * (df["X3"] + df["X4"] + df["epsilon_0"])
    df["Y(1)"] = (df["X2"] - df["X1"]) + 0.7 * (df["X3"] + df["X4"] + df["epsilon_1"])
    df["effect"] = df["Y(1)"] - df["Y(0)"]
    return df


def logistic(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def add_propensity_score(df: pd.DataFrame) -> pd.DataFrame:
    # check if X1, X2, X3, X4 are in the DataFrame
    if not all([col in df.columns for col in ["X1", "X2", "X3", "X4"]]):
        raise ValueError("DataFrame must contain X1, X2, X3, and X4 columns")
    df["p_score"] = logistic(
        np.log(0.5)
        + (np.log(2) - np.log(0.5)) * (df["X1"] + df["X2"] + df["X3"] + df["X4"]) / 4
    )
    return df


def add_treatment(df: pd.DataFrame) -> pd.DataFrame:
    # check if p_score is in the DataFrame
    if "p_score" not in df.columns:
        raise ValueError("DataFrame must contain p_score column")
    df["D"] = np.random.binomial(1, df["p_score"])
    return df


def generate_outcome(df: pd.DataFrame) -> pd.DataFrame:
    # check if D, Y(0), Y(1) are in the DataFrame
    if not all([col in df.columns for col in ["D", "Y(0)", "Y(1)"]]):
        raise ValueError("DataFrame must contain D, Y(0), and Y(1) columns")
    df["Y"] = df["D"] * df["Y(1)"] + (1 - df["D"]) * df["Y(0)"]
    return df


def generate_data(
    n: int,
    error_variance: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a DataFrame with independent features X1, X2, X3, and X4,
    potential outcomes Y(0) and Y(1), propensity score, treatment D, and observed outcome Y.

    Args:
        n (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    df = generate_x(n)
    df = add_potential_outcomes(df, error_variance)
    df = add_propensity_score(df)
    df = add_treatment(df)
    df = generate_outcome(df)
    return df


if __name__ == "__main__":
    import importlib
    import utils.visualization as viz

    importlib.reload(viz)

    np.random.seed(0)

    # set parameters
    n = 500
    error_variance = 0.2

    # generate data
    df = generate_data(
        n,
        error_variance=error_variance,
    )

    # print data summary
    print(df.describe())

    # visualize data
    fig_p_score = viz.scatter(
        df,
        x_col="X1",
        y_col="X2",
        color_col="p_score",
    )

    fig_effect = viz.scatter(
        df,
        x_col="X1",
        y_col="X2",
        color_col="effect",
    )
