import numpy as np
import pandas as pd


class DiscretizationSolverFactory:

    def __init__(self, col_1, col_2, col_1_range, col_2_range) -> None:  # type: ignore
        self.col_1 = col_1
        self.col_2 = col_2
        self.col_1_range = col_1_range
        self.col_2_range = col_2_range

    def create_model(self):
        return DiscretizationSolver(
            self.col_1,
            self.col_2,
            self.col_1_range,
            self.col_2_range,
        )


class DiscretizationSolver:
    def __init__(self, col_1, col_2, col_1_range, col_2_range, treat_empty_cell=False):
        self.col_1 = col_1
        self.col_2 = col_2
        self.col_1_left, self.col_1_right = col_1_range
        self.col_2_left, self.col_2_right = col_2_range
        self.trained = False
        self.treat_empty_cell = treat_empty_cell

    def fit(self, data: pd.DataFrame, covariate_cols, reward_col, **kwarg):
        grid_n = kwarg.get("grid_n", 10)

        if self.col_1 not in covariate_cols or self.col_2 not in covariate_cols:
            raise ValueError(f"{self.col_1} and {self.col_2} must be in covariate_cols")

        self.data = data.copy()

        # Create bins
        col1_bins = np.linspace(self.col_1_left, self.col_1_right, grid_n + 1)
        col2_bins = np.linspace(self.col_2_left, self.col_2_right, grid_n + 1)

        # Digitize data points into grid cells
        col1_idx = np.digitize(data[self.col_1], col1_bins) - 1
        col2_idx = np.digitize(data[self.col_2], col2_bins) - 1

        # Clamp indices to grid
        col1_idx = np.clip(col1_idx, 0, grid_n - 1)
        col2_idx = np.clip(col2_idx, 0, grid_n - 1)

        # Create reward grid
        reward_grid = np.full((grid_n, grid_n), np.nan)
        count_grid = np.zeros((grid_n, grid_n))

        for i in range(len(data)):
            r = col1_idx[i]
            c = col2_idx[i]
            if np.isnan(reward_grid[r][c]):
                reward_grid[r][c] = data[reward_col].iloc[i]
            else:
                reward_grid[r][c] += data[reward_col].iloc[i]
            count_grid[r][c] += 1

        # Average rewards
        reward_grid = reward_grid / count_grid

        self.reward_grid = reward_grid
        self.count_grid = count_grid
        self.col1_bins = col1_bins
        self.col2_bins = col2_bins
        self.grid_n = grid_n
        self.trained = True

        self.data = self.apply_policy(self.data, "z")
        self.objective_value = (self.data[reward_col] * self.data["z"]).sum()
        self.status = "Trained"

    def get_results(self):
        return {
            "status": self.status,
            "trained": self.trained,
            "z": self.data["z"],
            "objective_value": self.objective_value,
        }

    def apply_policy(
        self, data: pd.DataFrame, assignment_col="assignment"
    ) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("You must call `fit()` before applying the policy.")

        # Digitize new data points
        col1_idx = np.digitize(data[self.col_1], self.col1_bins) - 1
        col2_idx = np.digitize(data[self.col_2], self.col2_bins) - 1

        # Clamp indices to valid grid range
        col1_idx = np.clip(col1_idx, 0, self.grid_n - 1)
        col2_idx = np.clip(col2_idx, 0, self.grid_n - 1)

        assignments = []
        for r, c in zip(col1_idx, col2_idx):
            avg_reward = self.reward_grid[r][c]
            if np.isnan(avg_reward):
                decision = 1 if self.treat_empty_cell else 0
            elif avg_reward > 0:
                decision = 1
            else:
                decision = 0
            assignments.append(decision)

        # Add assignments to a copy of the DataFrame
        result = data.copy()
        result[assignment_col] = assignments
        return result


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    n = 50
    num_features = 2
    beta_min = -100
    beta_max = 100
    seed = 123
    np.random.seed(seed)

    # beta_true = np.array([0.8, -1.0, -1.0])  # True coefficients for
    beta_true = np.array([0.2, 1.0, -1.0])  # True coefficients for

    # Generate synthetic data
    X = np.random.uniform(0, 1, size=(n, num_features))
    # add a column of ones for intercept
    X = np.hstack((np.ones((n, 1)), X))

    noise = np.random.normal(0, 0.2, size=n)
    g = X @ beta_true + noise

    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(num_features + 1)])
    df["reward"] = g
    df["reward_positive"] = df["reward"] >= 0

    solver = DiscretizationSolver("X1", "X2", (0.2, 0.8), (0.2, 0.8))
    solver.fit(df, ["X1", "X2"], "reward", grid_n=5)

    # model = LinearThreshold(
    #     beta_min=beta_min, beta_max=beta_max, num_features=num_features
    # )
    # model.fit(
    #     df,
    #     covariate_cols=[f"X{i+1}" for i in range(num_features)],
    #     reward_col="reward",
    # )

    # results = model.get_results()

    # results["beta"]  # Coefficients

    # import matplotlib.pyplot as plt

    # df_plot = model.data

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     df_plot["X1"],
    #     df_plot["X2"],
    #     c=df_plot["reward_positive"],
    #     edgecolors="w",
    # )
    # # add a line with beta
    # x_vals = np.linspace(0, 1, 100)
    # y_vals = -beta_true[0] / beta_true[2] - x_vals * (beta_true[1] / beta_true[2])
    # ax.plot(x_vals, y_vals, color="red", label="True Line")
    # ax.set_xlabel("X1")
    # ax.set_ylabel("X2")
    # ax.set_title("Scatter Plot with True Line")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.legend()
    # fig.show()

    # fig, ax = plt.subplots()
    # ax.scatter(
    #     df_plot["X1"],
    #     df_plot["X2"],
    #     c=df_plot["z"],
    #     edgecolors="w",
    # )
    # # add a line with beta
    # x_vals = np.linspace(0, 1, 100)
    # y_vals = -(model.beta[0].value() / model.beta[2].value()) - x_vals * (  # type: ignore
    #     model.beta[1].value() / model.beta[2].value()
    # )  # type: ignore
    # ax.plot(x_vals, y_vals, color="red", label="Policy Line")
    # ax.set_xlabel("X1")
    # ax.set_ylabel("X2")
    # ax.set_title("Scatter Plot with Policy Line")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.legend()
    # fig.show()
