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
effect_shape = "ellipse"
plot_effect(effect_shape=effect_shape)
n_regret_calculation = 100
optimal_welfare = calculate_optimal_welfare(
    effect_shape=effect_shape, n=n_regret_calculation
)
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
        k_range=range(5, 16, 1),
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

# Create DataFrame
df_results = pd.DataFrame(results)

# Group by sample_size and calculate averages
regret_cols = ["regret_adaptive"] + [f"regret_{k}" for k in candidate_k]
avg_regrets = df_results.groupby("sample_size")[regret_cols].mean()

# Calculate winner shares
winner_shares = (
    df_results.groupby("sample_size")["winner"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)


def plot_regrets():
    plt.figure(figsize=(10, 6))
    for col in avg_regrets.columns:
        plt.plot(avg_regrets.index, avg_regrets[col], marker="o", label=col)
    plt.xlabel("Sample Size")
    plt.ylabel("Average Regret")
    plt.legend()
    plt.title("Average Regret by Sample Size")
    plt.show()


plot_regrets()


def plot_winner_shares():
    plt.figure(figsize=(10, 6))
    winner_shares.plot(kind="area", stacked=True, ax=plt.gca())
    plt.xlabel("Sample Size")
    plt.ylabel("Share")
    plt.title("Winner Shares by Sample Size")
    plt.legend(title="Winner K", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


plot_winner_shares()
