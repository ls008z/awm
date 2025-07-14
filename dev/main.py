from dev.awm.draft_awm import AdaptiveWelfareMaximization
from dev.milp.linear_threshold import LnearThresholdSolverFactory
from dev.estimation.estimators import (
    ConditionalMeanModelFactory,
    PropensityScoreModelFactory,
)
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import utils.visualization as viz
from dev.dgp.hte import (
    calculate_optimal_welfare,
    calculate_regret,
    calculate_regret_integrated,
    generate_data,
    generate_meshgrid_dataframe,
    plot_effect,
    plot_policy,
)


df_dgp = generate_data(n=200, error_variance=0.2)
plot_effect()
viz.scatter(df_dgp, x_col="X1", y_col="X2", s=10, color_col="effect")


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
    beta_max=10, beta_min=-10, num_features=5
)
awm.empirical_walfare_maximization(
    empirical_welfare_maximizor_factory=linear_threshold_factory,
)

welfare = awm.calculate_ooo_welfare()

print(f"CV welfare {welfare}")

print("calculating final emw")
solver = awm.calculate_final_ewm(linear_threshold_factory)
print(f"Final fit in sample welfare {solver.objective_value}")  # type: ignore
manueal_check = (solver.data["z"] * solver.data["tau"]).sum()  # type: ignore

plot_policy(solver)

optimal_welfare = calculate_optimal_welfare(5000)
print(f"optimal welfare {optimal_welfare}")
regret = calculate_regret_integrated(
    awm_solver=solver, optimal_welfare=optimal_welfare, n=1000
)
print(f"regret {regret}")
