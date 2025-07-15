import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dev.dgp.hte import (
    generate_data,
    plot_effect,
    calculate_optimal_welfare,
    calculate_regret_integrated,
)
from dev.awm.awm import run_awm_grid_size_selection
from tqdm import tqdm
from dev.milp.discretize import DiscretizationSolver


np.random.seed(42)
n_monte_carlo = 200
list_sample_size = [
    # 100,
    200,
    # 300,
    400,
    # 500,
    600,
    # 700,
    800,
    # 900,
    1000,
    1200,
    1400,
    1600,
    # 1800,
]
effect_shape = "rectangle"
plot_effect(effect_shape=effect_shape)
n_regret_calculation = 100
optimal_welfare = calculate_optimal_welfare(
    effect_shape=effect_shape, n=n_regret_calculation
)
candidate_k = [3, 5, 10]
candidate_k = [5, 10, 15]


# Initialize results DataFrame
results = []

for sample_size, m_sim in tqdm(
    [(s, m) for s in list_sample_size for m in range(n_monte_carlo)],
    desc="Running simulations",
    total=len(list_sample_size) * n_monte_carlo,
):  # print(f"sample_size: {sample_size}, m_sim: {m_sim}")

    df_dgp = generate_data(n=sample_size, error_variance=0.2, effect_shape=effect_shape)

    covariates = ["X1", "X2", "X3", "X4"]
    outcome = "Y"
    treatment = "D"
    df_observed = df_dgp[covariates + [outcome, treatment]].copy()

    adaptive = run_awm_grid_size_selection(
        data=df_observed,
        covariate_cols=covariates,
        outcome_col=outcome,
        treatment_col=treatment,
        k_range=range(3, 11, 1),
        verbose=False,
    )

    regret_adaptive = calculate_regret_integrated(
        awm_solver=adaptive.final_solution,
        optimal_welfare=optimal_welfare,
        effect_shape=effect_shape,
        n=n_regret_calculation,
    )
    winner = adaptive.final_solution.grid_n

    train_data = adaptive.provide_final_training_data()

    # Store regrets for each k
    regret_k = {}
    for k in candidate_k:
        empirical_welfare_maximizor = DiscretizationSolver(
            col_1="X1", col_2="X2", col_1_range=(0, 1), col_2_range=(0, 1)
        )

        empirical_welfare_maximizor.fit(
            train_data,
            covariate_cols=covariates,
            reward_col="tau",
            grid_n=k,
        )

        regret_k[f"regret_{k}"] = calculate_regret_integrated(
            awm_solver=empirical_welfare_maximizor,
            optimal_welfare=optimal_welfare,
            effect_shape=effect_shape,
            n=n_regret_calculation,
        )

    # Record results
    result = {
        "sample_size": sample_size,
        "m_sim": m_sim,
        "regret_adaptive": regret_adaptive,
        "winner": winner,
        **regret_k,
    }
    results.append(result)

# # Create DataFrame
df_results = pd.DataFrame(results)

# df_results = pd.read_csv("data/results/ellipse.csv")

# Group by sample_size and calculate averages
regret_cols = ["regret_adaptive"] + [f"regret_{k}" for k in candidate_k]
avg_regrets = df_results.groupby("sample_size")[regret_cols].mean()

# Calculate winner shares
winner_shares = (
    df_results.groupby("sample_size")["winner"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_winner_shares_greyscale():
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Define hatch patterns
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    # Generate a modern, low-contrast color palette
    num_layers = len(winner_shares.columns)
    cmap = cm.get_cmap("tab20c")  # Use a muted, modern palette
    colors = [cmap(i / num_layers) for i in range(num_layers)]

    # Plot each column individually with color and hatch
    bottom = None
    for i, column in enumerate(winner_shares.columns):
        top = winner_shares[column] + (bottom if bottom is not None else 0)

        ax.fill_between(
            winner_shares.index,
            top,
            bottom if bottom is not None else 0,
            label=column,
            hatch=hatches[i % len(hatches)],
            edgecolor="black",
            facecolor=colors[i % len(colors)],
            linewidth=0.5,
        )
        bottom = top

    plt.xlabel("Sample Size")
    plt.ylabel("Share")
    plt.ylim(0, 1)
    plt.title("Shares of Complexity Chosen by AWM by Sample Size")

    # Reverse and apply legend with larger handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="Complexity",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        handlelength=3,
        handleheight=2.0,
        handletextpad=0.5,
        borderaxespad=0.5,
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.show()


plot_winner_shares_greyscale()
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_regrets():
    plt.figure(figsize=(10, 6))

    # Use modern color palette
    num_lines = len(avg_regrets.columns)
    cmap = get_cmap("Dark2")
    colors = [cmap(i / num_lines) for i in range(num_lines)]

    # Define a list of distinct markers
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "+", "x"]

    # Build the rename dictionary
    rename_dict = {"regret_adaptive": "AWM"}

    # Dynamically add the "regret_{i}" → "EWM at Complexity {i}" mappings
    for col in avg_regrets.columns:
        if col.startswith("regret_") and col != "regret_adaptive":
            try:
                # Extract the integer i after 'regret_'
                i = int(col.split("_")[1])
                rename_dict[col] = f"EWM at Complexity {i}"
            except ValueError:
                pass  # Ignore any that don't follow the expected pattern

    # Apply the renaming
    avg_regrets.rename(columns=rename_dict, inplace=True)

    for i, col in enumerate(avg_regrets.columns):
        plt.plot(
            avg_regrets.index,
            avg_regrets[col],
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=col,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Sample Size")
    plt.ylabel("Average Regret")
    plt.title("Average Regret by Sample Size")
    plt.legend(
        title="Complexity", fontsize=10, title_fontsize=11, loc="best", frameon=False
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


plot_regrets()
