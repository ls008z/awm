from dev.awm.core_awm import AdaptiveWelfareMaximization
from dev.milp.linear_threshold import LnearThresholdSolverFactory
from dev.milp.discretize import DiscretizationSolverFactory
from dev.estimation.estimators import (
    ConditionalMeanModelFactory,
    PropensityScoreModelFactory,
)
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import utils.visualization as viz
from dev.dgp.hte import (
    calculate_optimal_welfare,
    calculate_regret_integrated,
    generate_data,
    generate_meshgrid_dataframe,
    plot_effect,
    plot_policy,
)

effect_shape = "ellipse"

df_dgp = generate_data(n=100, error_variance=0.2, effect_shape=effect_shape)
num_features = 9
grid_n = 20

plot_effect(effect_shape=effect_shape)
viz.scatter(df_dgp, x_col="X1", y_col="X2", s=5, color_col="effect")


covariates = ["X1", "X2", "X3", "X4"]
outcome = "Y"
treatment = "D"

poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(df_dgp[covariates])
feature_names = poly.get_feature_names_out(covariates)
df_poly = pd.DataFrame(poly_features, columns=feature_names)
df_observed = pd.concat([df_poly, df_dgp[[outcome, treatment]]], axis=1)

covariates_poly = [
    "X1",
    "X1^2",
    "X2",
    "X2^2",
    "X1 X2",
    # ----------
    "X3",
    "X3^2",
    "X1 X3",
    "X2 X3",
    "X4",
    "X4^2",
    "X1 X4",
    "X2 X4",
    "X3 X4",
    # ----------
    "X1^3",
    "X1^2 X2",
    "X2^3",
    "X1 X2^2",
    # ----------
    "X1^2 X3",
    "X1^2 X4",
    "X1 X2 X3",
    "X1 X2 X4",
    "X1 X3^2",
    "X1 X3 X4",
    "X1 X4^2",
    "X2^2 X3",
    "X2^2 X4",
    "X2 X3^2",
    "X2 X3 X4",
    "X2 X4^2",
    "X3^3",
    "X3^2 X4",
    "X3 X4^2",
    "X4^3",
]

covariates_x1_x2 = [
    "X1",
    "X1^2",
    "X2",
    "X2^2",
    "X1 X2",
    # ----------
    "X1^3",
    "X1^2 X2",
    "X2^3",
    "X1 X2^2",
    # ----------
    "X3",
    "X4",
]

awm = AdaptiveWelfareMaximization(
    df_observed,
    covariates_x1_x2,
    outcome,
    treatment,
)

awm.cv_sample_splitting(cross_vailidation_folds=4, seed=0)
awm.create_cross_fitting_data(cross_fitting_folds=5, seed=0)


p_score_model_factory = PropensityScoreModelFactory(model_type="logistic")
conditional_mean_model_factory = ConditionalMeanModelFactory(model_type="OLS")

awm.estimate_all_nussance_parameters(
    p_score_model_factory=p_score_model_factory,
    conditional_mean_model_factory=conditional_mean_model_factory,
)


linear_threshold_factory = LnearThresholdSolverFactory(
    beta_max=10, beta_min=-10, num_features=num_features
)

discretization_solver_factory = DiscretizationSolverFactory(
    col_1="X1", col_2="X2", col_1_range=(0, 1), col_2_range=(0, 1)
)

# awm.empirical_walfare_maximization(
#     empirical_welfare_maximizor_factory=linear_threshold_factory,
#     emw_covariate_cols=None,
#     # grid_n=grid_n,
# )

# welfare = awm.calculate_ooo_welfare()

# print(f"CV welfare {welfare}")

print("calculating final emw")
solver = awm.calculate_final_ewm(
    linear_threshold_factory,
    # discretization_solver_factory,
    # grid_n=grid_n,
)


print(f"Final fit in sample welfare {solver.objective_value}")  # type: ignore
# manueal_check = (solver.data["z"] * solver.data["tau"]).sum()  # type: ignore

plot_policy(
    solver,
    effect_shape=effect_shape,
    title="Linear Threshold Rules on Ellipse DGP",
    # title="Discretized Rules on Ellipse DGP",
)

# optimal_welfare = calculate_optimal_welfare(effect_shape=effect_shape, n=5000)
# print(f"optimal welfare {optimal_welfare}")
# regret = calculate_regret_integrated(
#     awm_solver=solver,
#     optimal_welfare=optimal_welfare,
#     n=5000,
#     effect_shape=effect_shape,
# )
# print(f"regret {regret}")
