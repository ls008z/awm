import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


n = 100
num_features = 2
beta_min = -100
beta_max = 100
seed = 123
np.random.seed(seed)

beta_true = np.array([0.8, -1.0, -1.0])  # True coefficients for
beta_true = np.array([0.2, 1.0, -1.0])  # True coefficients for

# Generate synthetic data
X = np.random.uniform(0, 1, size=(n, num_features))
# add a column of ones for intercept
X = np.hstack((np.ones((n, 1)), X))

noise = np.random.normal(0, 0.5, size=n)
g = X @ beta_true + noise

df = pd.DataFrame(X, columns=[f"X{i}" for i in range(num_features + 1)])
df["reward"] = g
df["reward_positive"] = df["reward"] >= 0

fig, ax = plt.subplots()
ax.scatter(
    df["X1"],
    df["X2"],
    c=df["reward_positive"],
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


# Compute tight Big-M
M = max(abs(beta_min), abs(beta_max)) * np.max(np.abs(X), axis=0).sum() * 2
epsilon = 1e-6  # Small value for strict inequality approximation

# Initialize model
model = pulp.LpProblem("MILP_Indicator_Model", pulp.LpMaximize)

# Define variables
beta = [
    pulp.LpVariable(f"beta_{j}", lowBound=beta_min, upBound=beta_max)
    for j in range(num_features + 1)
]
z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]
t = [pulp.LpVariable(f"t_{i}") for i in range(n)]

# Objective function
model += pulp.lpSum(g[i] * z[i] for i in range(n))

# Constraints
for i in range(n):
    model += (
        t[i] == pulp.lpSum(X[i, j] * beta[j] for j in range(num_features + 1)),
        f"dot_prod_{i}",
    )
    model += t[i] >= -M * (1 - z[i]), f"z_ub_{i}"
    model += t[i] <= M * (z[i] - epsilon), f"z_lb_{i}"


# Solve
model.solve(pulp.PULP_CBC_CMD(msg=False))

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Objective: {pulp.value(model.objective):.4f}")
print(f"Big-M: {M:.2f}")
print(f"beta: {[v.value() for v in beta]}")

# calculate agree ratio
z_positive = [v.value() > 0 for v in z]
g_positive = [g[i] > 0 for i in range(n)]
agree = [z_positive[i] == g_positive[i] for i in range(n)]
agree_ratio = sum(agree) / n
print(f"Agree Ratio: {agree_ratio:.2f}")


df["z"] = [v.value() for v in z]
df["t_imputed"] = (
    df["X0"] * beta[0].value() + df["X1"] * beta[1].value() + df["X2"] * beta[2].value()
)
df["t"] = [v.value() for v in t]

fig, ax = plt.subplots()
ax.scatter(
    df["X1"],
    df["X2"],
    c=df["z"],
    edgecolors="w",
)
# add a line with beta
x_vals = np.linspace(0, 1, 100)
y_vals = -(beta[0].value() / beta[2].value()) - x_vals * (
    beta[1].value() / beta[2].value()
)
ax.plot(x_vals, y_vals, color="red", label="Policy Line")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("Scatter Plot with Policy Line")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
fig.show()
