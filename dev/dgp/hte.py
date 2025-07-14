# generate data (Y, X1, X2, X3, X4, D, Y(1), Y(0))
# E[Y(1)-Y(0)|X] > 0 iff X2 > X1

import numpy as np
import pandas as pd
import utils.visualization as viz
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import itertools


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


def generate_meshgrid_dataframe(n_points=10, d=2):
    # Create linearly spaced values for each axis
    axes = [np.linspace(0, 1, n_points) for _ in range(d)]

    # Cartesian product of the axes to get grid points
    mesh = list(itertools.product(*axes))

    # Create column names: X1, X2, ..., Xd
    col_names = [f"X{i+1}" for i in range(d)]

    # Convert to DataFrame
    df = pd.DataFrame(mesh, columns=col_names)
    return df


def add_potential_outcomes_diagnal_effect(
    df: pd.DataFrame,
    error_variance: float = 1.0,
) -> pd.DataFrame:
    df["epsilon_0"] = np.random.normal(0, error_variance, len(df))
    df["epsilon_1"] = np.random.normal(0, error_variance, len(df))
    df["Y(0)"] = 0.7 * (df["X3"] + df["X4"] + df["epsilon_0"])
    df["Y(1)"] = (df["X2"] - df["X1"]) + 0.7 * (df["X3"] + df["X4"] + df["epsilon_1"])
    df["effect"] = df["Y(1)"] - df["Y(0)"]
    return df


def add_ellipse_hte(
    df: pd.DataFrame,
):
    ellipse_input = (
        15.625 * df["X1"] ** 2
        - 18.75 * df["X1"] * df["X2"]
        - 6.25 * df["X1"]
        + 15.625 * df["X2"] ** 2
        - 6.25 * df["X2"]
        + 2.125
    )

    ellipse_input = ellipse_input.clip(lower=-1, upper=1)

    df["effect_mean"] = -ellipse_input

    return df


def add_potential_outcomes_ellipse_effect(
    df: pd.DataFrame,
    error_variance: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()  # avoid modifying original DataFrame
    df["epsilon_0"] = np.random.normal(0, error_variance, len(df))
    df["epsilon_1"] = np.random.normal(0, error_variance, len(df))

    df["Y(0)"] = 0.7 * (df["X3"] + df["X4"] + df["epsilon_0"])

    df = add_ellipse_hte(df)

    df["Y(1)"] = df["effect_mean"] + 0.7 * (df["X3"] + df["X4"] + df["epsilon_1"])

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
    df = add_potential_outcomes_ellipse_effect(df, error_variance)
    df = add_propensity_score(df)
    df = add_treatment(df)
    df = generate_outcome(df)
    return df


def plot_effect():
    df = generate_meshgrid_dataframe(100)
    df = add_ellipse_hte(df)

    # Create scatter plot using matplotlib directly
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(df["X1"], df["X2"], c=df["effect_mean"], cmap="inferno", s=10)
    ax.set_title("CATE and Optimal Policy")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="effect")

    # Parameters for the tilted ellipse
    h, k = 0.5, 0.5  # center
    a, b = 0.4, 0.2  # axes
    theta = np.pi / 4  # rotation angle (45 degrees)

    # Create rotated ellipse mask
    x = np.linspace(0, 1, 800)
    y = np.linspace(0, 1, 800)
    X, Y = np.meshgrid(x, y)

    X_rot = (X - h) * np.cos(theta) + (Y - k) * np.sin(theta)
    Y_rot = -(X - h) * np.sin(theta) + (Y - k) * np.cos(theta)
    Z = (X_rot / a) ** 2 + (Y_rot / b) ** 2

    # Add contour for the ellipse
    ax.contour(X, Y, Z, levels=[1], colors="white")

    ax.grid(True)
    plt.show()


def plot_policy(awm_solver, n=100):
    df = generate_meshgrid_dataframe(n)
    df["X3"] = 0.5
    df["X4"] = 0.5

    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)

    # replace them with E[X_3^2] and E[X_4^2]
    df = pd.DataFrame(poly_features, columns=feature_names)
    df["X_3^2"] = 1 / 3
    df["X_4^2"] = 1 / 3

    df = awm_solver.apply_threshold(df)

    # Color and label mapping
    color_map = {0: "black", 1: "grey"}
    label_map = {0: "Leave out", 1: "Treat"}

    # Map colors
    colors = df["assignment"].map(color_map)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(df["X1"], df["X2"], c=colors, s=10)

    # Add dummy scatter points for legend
    for value in [0, 1]:
        ax.scatter([], [], c=color_map[value], label=label_map[value], s=30)

    # Labels and styling
    ax.set_title("Policy")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(title="Assignment")

    # Parameters for the tilted ellipse
    h, k = 0.5, 0.5  # center
    a, b = 0.4, 0.2  # axes
    theta = np.pi / 4  # rotation angle (45 degrees)

    # Create rotated ellipse mask
    x = np.linspace(0, 1, 800)
    y = np.linspace(0, 1, 800)
    X, Y = np.meshgrid(x, y)

    X_rot = (X - h) * np.cos(theta) + (Y - k) * np.sin(theta)
    Y_rot = -(X - h) * np.sin(theta) + (Y - k) * np.cos(theta)
    Z = (X_rot / a) ** 2 + (Y_rot / b) ** 2

    # Add contour for the ellipse
    ax.contour(X, Y, Z, levels=[1], colors="white")

    ax.grid(True)
    plt.show()


def calculate_optimal_welfare(n=100):
    df = generate_meshgrid_dataframe(n_points=n)
    df = add_ellipse_hte(df)
    df["optimal_assignment"] = 1 * (df["effect_mean"] > 0)
    avg_welfare = (df["optimal_assignment"] * df["effect_mean"]).sum() / len(df)
    return avg_welfare


def calculate_regret(awm_solver, optimal_welfare, n=10):
    df = generate_meshgrid_dataframe(n_points=n, d=4)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)
    df = pd.DataFrame(poly_features, columns=feature_names)
    df = add_ellipse_hte(df)
    df = awm_solver.apply_threshold(df)
    avg_welfare = (df["assignment"] * df["effect_mean"]).sum() / len(df)
    regret = avg_welfare - optimal_welfare
    return regret


def calculate_regret_integrated(awm_solver, optimal_welfare, n=100):
    df = generate_meshgrid_dataframe(n_points=n, d=2)
    df["X3"] = 0.5
    df["X4"] = 0.5

    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)

    df = pd.DataFrame(poly_features, columns=feature_names)

    # replace them with E[X_3^2] and E[X_4^2]
    df["X_3^2"] = 1 / 3
    df["X_4^2"] = 1 / 3

    df = add_ellipse_hte(df)
    df = awm_solver.apply_threshold(df)
    avg_welfare = (df["assignment"] * df["effect_mean"]).sum() / len(df)
    regret = avg_welfare - optimal_welfare
    return regret


if __name__ == "__main__":
    import importlib
    import utils.visualization as viz

    importlib.reload(viz)

    np.random.seed(0)

    # set parameters
    n = 1000
    error_variance = 0.5

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

    # ----------------------------------------------------------------------
    # Calculate optimal regret
    # ----------------------------------------------------------------------

    import time

    optimal_welfare = {}
    for n in [100, 1000, 5000, 10000]:
        start_time = time.time()
        welfare = calculate_optimal_welfare(n)
        elapsed_time = time.time() - start_time
        print(f"optimal_walfare with n = {n}: {welfare}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        optimal_welfare[n] = welfare

    # df["effect"].hist(bins=30)

    # df_grid = generate_meshgrid_dataframe(100)

    # def apply_ellipse_rule(df):
    #     df["assignment"] = (
    #         1
    #         * (
    #             15.625 * df["X1"] ** 2
    #             - 18.75 * df["X1"] * df["X2"]
    #             - 6.25 * df["X1"]
    #             + 15.625 * df["X2"] ** 2
    #             - 6.25 * df["X2"]
    #             + 2.125
    #         )
    #         >= 0
    #     )

    # apply_ellipse_rule(df_grid)

    # fig_assigment = viz.scatter(
    #     df_grid,
    #     x_col="X1",
    #     y_col="X2",
    #     color_col="assignment",
    # )
