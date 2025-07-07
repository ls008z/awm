import pulp
import numpy as np
import pandas as pd


def solve_big_m_milp(
    n,
    beta_true,
    d=3,
    beta_min=-10,
    beta_max=10,
    seed=42,
):
    # Generate synthetic data
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(n, d))

    noise = np.random.normal(0, 10, size=n)
    g = X @ beta_true + noise

    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])
    df["reward"] = g

    # Compute tight Big-M
    M = max(abs(beta_min), abs(beta_max)) * np.max(np.abs(X), axis=0).sum() * 2

    # Initialize model
    model = pulp.LpProblem("MILP_Indicator_Model", pulp.LpMaximize)

    # Define variables
    beta = [
        pulp.LpVariable(f"beta_{j}", lowBound=beta_min, upBound=beta_max)
        for j in range(d)
    ]
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]
    t = [pulp.LpVariable(f"t_{i}") for i in range(n)]

    # Objective function
    model += pulp.lpSum(g[i] * z[i] for i in range(n))

    # Constraints
    for i in range(n):
        model += (
            t[i] == pulp.lpSum(X[i, j] * beta[j] for j in range(d)),
            f"dot_prod_{i}",
        )
        model += t[i] >= -M * (1 - z[i]), f"bigM_lb_{i}"
        model += t[i] <= M * z[i], f"bigM_ub_{i}"

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "beta": [v.value() for v in beta],
        "z": [v.value() for v in z],
        "big_m": M,
        "g": g,
        "df": df,
    }


# Example usage
if __name__ == "__main__":

    n = 20
    d = 2
    beta_min = -10
    beta_max = 10
    seed = 123
    beta_true = np.random.uniform(-1, 1, size=d)

    result = solve_big_m_milp(n=n, beta_true=beta_true, d=d)

    print(f"Status: {result['status']}")
    print(f"Objective: {result['objective']:.4f}")
    print(f"Big-M: {result['big_m']:.2f}")

    beta = result["beta"]
    normalized_beta = [float(b * np.mean(beta_true) / np.mean(beta)) for b in beta]
    print(f"beta: {beta}")
    print(f"beta true: {[f'{b:.6f}' for b in beta_true]}")
    print(f"normalized {[f'{b:.6f}' for b in normalized_beta]}")

    z_binary = [i == 1.0 for i in result["z"]]
    g_binary = [bool(i >= 0) for i in result["g"]]
    agree_ratio = sum([i == j for i, j in zip(z_binary, g_binary)]) / n
    print(f"agree_ratio: {agree_ratio:.2f}")

    # print(f"z: {z_binary}")
    # print(f"g: {g_binary}")

    import matplotlib.pyplot as plt

    df = result["df"]
    df["reward_positive"] = df["reward"] >= 0

    fig, ax = plt.subplots()
    ax.scatter(
        df["X0"],
        df["X1"],
        c=df["reward_positive"],
        edgecolors="w",
    )

    # add a line with beta
    x_vals = np.linspace(-1, 1, 100)
    y_vals = (beta[0] * x_vals + beta[1]) / np.mean(beta_true)
    ax.plot(x_vals, y_vals, color="red", label="Estimated Line")
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_title("Scatter Plot with Estimated Line")
    ax.legend()
