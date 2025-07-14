import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(123)

# Number of samples
n = 1000

# Generate covariates x uniformly in range [0, 10]
x = np.random.uniform(0, 10, n)

# Generate rewards r: can be random, or increasing with x to simulate correlation
# Option 1: random rewards
# r = np.random.uniform(0, 1, n)

# Option 2: increasing with x + noise
r = 0.5 * (x - 4) + np.random.normal(0, 1, n)

# Combine into DataFrame
df = pd.DataFrame({"x": x, "r": r})

# quick plot
plt.scatter(df["x"], df["r"])
plt.xlabel("x")
plt.ylabel("r")


# Big-M constant
M = (max(x) - min(x) + 1) * 10
epsilon = 1e-5  # small positive constant

# Create the MILP problem
prob = pulp.LpProblem("Threshold_Max_Reward", pulp.LpMaximize)

# Variables
z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]
t = pulp.LpVariable("t", lowBound=min(x) - 1, upBound=max(x) + 1, cat="Continuous")

# Objective function: maximize total reward
prob += pulp.lpSum([z[i] * r[i] for i in range(n)])

# Constraints using Big-M to model z_i = 1{x_i >= t}
for i in range(n):
    prob += x[i] - t >= -M * (1 - z[i]), f"big_m_constraint_{i}"
    prob += x[i] - t <= M * (z[i] - epsilon), f"upper_{i}"

# ==== Step 3: Solve ====
prob.solve()

# ==== Step 4: Output Results ====
print("\nStatus:", pulp.LpStatus[prob.status])
print("Optimal threshold t:", t.varValue)
print("Total reward:", pulp.value(prob.objective))

import matplotlib.pyplot as plt

df["z"] = [v.varValue for v in z]

plt.figure(figsize=(10, 6))
plt.scatter(
    df["x"],
    df["r"],
    label="Data Points",
    c=df["z"],
)
plt.axvline(
    x=t.varValue,
    color="red",
    linestyle="--",
    label=f"Optimal Threshold: {t.varValue:.2f}",
)
# add a line at y=0
plt.axhline(0, color="black", linestyle="--")
