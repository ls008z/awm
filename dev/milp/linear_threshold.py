import pulp
import numpy as np


class LinearThreshold:
    def __init__(self, beta_min, beta_max, num_features):

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_features = num_features
        self.model = pulp.LpProblem("LinearThreshold", pulp.LpMaximize)
        self.epsilon = 1e-6  # Small value for strict inequality approximation

    def fit(self, data, covariate_cols, reward_col):

        self.data = data.copy()
        self.covariate_cols = covariate_cols
        self.reward_col = reward_col

        self.x = data[covariate_cols[: self.num_features]].values
        self.x = np.hstack((np.ones((self.x.shape[0], 1)), self.x))
        self.y = data[reward_col].values

        # Ensure the data is in float format
        self.x = self.x.astype(float)
        self.y = self.y.astype(float)

        # Define variables
        self.beta = [
            pulp.LpVariable(f"beta_{j}", lowBound=self.beta_min, upBound=self.beta_max)
            for j in range(self.num_features + 1)
        ]
        self.z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(len(data))]
        self.t = [pulp.LpVariable(f"t_{i}") for i in range(len(data))]

        # Objective function
        self.model += pulp.lpSum(self.y[i] * self.z[i] for i in range(len(data)))

        # Constraints
        for i in range(len(data)):
            self.model += (
                self.t[i]
                == pulp.lpSum(
                    self.x[i, j] * self.beta[j] for j in range(self.num_features + 1)
                ),
                f"dot_prod_{i}",
            )
            big_m = (
                max(abs(self.beta_min), abs(self.beta_max))
                * sum(abs(x) for x in self.x[i])
                * 2
            )
            self.model += self.t[i] >= -big_m * (1 - self.z[i]), f"bigM_lb_{i}"
            self.model += (
                self.t[i] <= big_m * (self.z[i] - self.epsilon),
                f"bigM_ub_{i}",
            )

        # Solve the model
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        self.status = pulp.LpStatus[self.model.status]
        self.objective_value = pulp.value(self.model.objective)
        self.beta_values = [v.value() for v in self.beta]
        self.z_values = [v.value() for v in self.z]

        # append values to the data

        self.data["t"] = [v.value() for v in self.t]
        self.data["z"] = self.z_values
        self.data["t_imputed"] = self.beta[0].value() + sum(  # type: ignore
            self.data[self.covariate_cols[j]] * self.beta_values[j + 1]
            for j in range(0, self.num_features)
        )

    def get_results(self):
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "beta": self.beta_values,
            "z": self.z_values,
            "t": [v.value() for v in self.t],
        }

    def apply_threshold(self, data, assignemt_col="assignment"):

        # check if model is at optimality
        if self.status != "Optimal":
            raise ValueError("Model is not at optimality. Please fit the model first.")
        # check covarate_cols in data
        if not all(col in data.columns for col in self.covariate_cols):
            raise ValueError("Covariate columns not found in the data.")

        x = data[self.covariate_cols[: self.num_features]].values
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        t = np.array(
            [
                sum(x[i, j] * self.beta_values[j] for j in range(self.num_features + 1))
                for i in range(len(data))
            ]
        )
        z = np.array([1 if t[i] > 0 else 0 for i in range(len(data))])
        data[assignemt_col] = z
        data["t"] = t
        return data


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    n = 100
    num_features = 2
    beta_min = -100
    beta_max = 100
    seed = 123
    np.random.seed(seed)

    beta_true = np.array([0.8, -1.0, -1.0])  # True coefficients for
    # beta_true = np.array([0.2, 1.0, -1.0])  # True coefficients for

    # Generate synthetic data
    X = np.random.uniform(0, 1, size=(n, num_features))
    # add a column of ones for intercept
    X = np.hstack((np.ones((n, 1)), X))

    noise = np.random.normal(0, 0.2, size=n)
    g = X @ beta_true + noise

    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(num_features + 1)])
    df["reward"] = g
    df["reward_positive"] = df["reward"] >= 0

    model = LinearThreshold(
        beta_min=beta_min, beta_max=beta_max, num_features=num_features
    )
    model.fit(
        df,
        covariate_cols=[f"X{i+1}" for i in range(num_features)],
        reward_col="reward",
    )

    results = model.get_results()

    results["beta"]  # Coefficients

    import matplotlib.pyplot as plt

    df_plot = model.data

    fig, ax = plt.subplots()
    ax.scatter(
        df_plot["X1"],
        df_plot["X2"],
        c=df_plot["reward_positive"],
        edgecolors="w",
    )
    # add a line with beta
    x_vals = np.linspace(0, 1, 100)
    y_vals = -beta_true[0] / beta_true[2] - x_vals * (beta_true[1] / beta_true[2])
    ax.plot(x_vals, y_vals, color="red", label="True Line")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Scatter Plot with True Line")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.scatter(
        df_plot["X1"],
        df_plot["X2"],
        c=df_plot["z"],
        edgecolors="w",
    )
    # add a line with beta
    x_vals = np.linspace(0, 1, 100)
    y_vals = -(model.beta[0].value() / model.beta[2].value()) - x_vals * (  # type: ignore
        model.beta[1].value() / model.beta[2].value()
    )  # type: ignore
    ax.plot(x_vals, y_vals, color="red", label="Policy Line")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Scatter Plot with Policy Line")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.show()
