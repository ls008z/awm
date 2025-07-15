from dev.awm.core_awm import AdaptiveWelfareMaximization
from dev.milp.linear_threshold import LnearThresholdSolverFactory
from dev.milp.discretize import DiscretizationSolverFactory
from dev.estimation.estimators import (
    ConditionalMeanModelFactory,
    PropensityScoreModelFactory,
)


def run_awm_covariate_selectoin(
    data, main_covariate_cols, dict_emw_covariate_cols, outcome_col, treatment_col
):
    awm = AdaptiveWelfareMaximization(
        data=data,
        covariate_cols=main_covariate_cols,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
    )
    awm.cv_sample_splitting(cross_vailidation_folds=4, seed=42)
    awm.create_cross_fitting_data(cross_fitting_folds=5, seed=0)

    p_score_model_factory = PropensityScoreModelFactory(model_type="logistic")
    conditional_mean_model_factory = ConditionalMeanModelFactory(model_type="OLS")
    awm.estimate_all_nussance_parameters(
        p_score_model_factory=p_score_model_factory,
        conditional_mean_model_factory=conditional_mean_model_factory,
    )

    linear_threshold_solver_factory = LnearThresholdSolverFactory(
        beta_max=10,
        beta_min=-10,
    )

    dict_cv_welfare = {}

    for key, emw_covariate_cols in dict_emw_covariate_cols.items():
        print(f"EMW with covariate cols: {emw_covariate_cols}")

        awm.empirical_walfare_maximization(
            empirical_welfare_maximizor_factory=linear_threshold_solver_factory,
            emw_covariate_cols=emw_covariate_cols,
        )

        welfare = awm.calculate_ooo_welfare()
        print(f"OOO welfare of {key}: {welfare}")

        dict_cv_welfare[key] = welfare

    # find the key with the highest welfare
    best_key = max(dict_cv_welfare, key=dict_cv_welfare.get)  # type: ignore

    print(f"winner is {best_key}, calaculating final ewm")

    awm.calculate_final_ewm(
        empirical_welfare_maximizor_factory=linear_threshold_solver_factory,
        emw_covariate_cols=dict_emw_covariate_cols[best_key],
    )

    print(f"status: {awm.final_solution.status}")
    print(f"optimal in-sample welfare {awm.final_solution.objective_value}")

    return awm


def run_awm_grid_size_selection(
    data,
    covariate_cols,
    outcome_col,
    treatment_col,
    k_range,
    verbose=True,
):
    awm = AdaptiveWelfareMaximization(
        data=data,
        covariate_cols=covariate_cols,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        verbose=verbose,
    )
    awm.cv_sample_splitting(cross_vailidation_folds=4, seed=42)
    awm.create_cross_fitting_data(cross_fitting_folds=5, seed=0)

    p_score_model_factory = PropensityScoreModelFactory(model_type="logistic")
    conditional_mean_model_factory = ConditionalMeanModelFactory(model_type="OLS")
    awm.estimate_all_nussance_parameters(
        p_score_model_factory=p_score_model_factory,
        conditional_mean_model_factory=conditional_mean_model_factory,
    )

    discretization_solver_factory = DiscretizationSolverFactory(
        col_1="X1", col_2="X2", col_1_range=(0, 1), col_2_range=(0, 1)
    )

    dict_cv_welfare = {}

    for k in k_range:
        if verbose:
            print(f"EMW with gird_n: {k}")

        awm.empirical_walfare_maximization(
            empirical_welfare_maximizor_factory=discretization_solver_factory, grid_n=k
        )

        welfare = awm.calculate_ooo_welfare()
        if verbose:
            print(f"OOO welfare of grid_n={k}: {welfare}")

        dict_cv_welfare[k] = welfare

    # find the key with the highest welfare
    optimal_k = max(dict_cv_welfare, key=dict_cv_welfare.get)  # type: ignore

    if verbose:
        print(f"winner is {optimal_k}, calaculating final ewm")

    awm.calculate_final_ewm(
        empirical_welfare_maximizor_factory=discretization_solver_factory,
        grid_n=optimal_k,
    )

    if verbose:
        print(f"status: {awm.final_solution.status}")
        print(f"optimal in-sample welfare {awm.final_solution.objective_value}")

    return awm


if __name__ == "__main__":

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

    df_dgp = generate_data(n=10000, error_variance=0.2, effect_shape=effect_shape)
    plot_effect(effect_shape=effect_shape)
    viz.scatter(df_dgp, x_col="X1", y_col="X2", s=10, color_col="effect")

    covariates = ["X1", "X2", "X3", "X4"]
    outcome = "Y"
    treatment = "D"
    df_observed = df_dgp[covariates + [outcome, treatment]].copy()

    ans = run_awm_grid_size_selection(
        data=df_observed,
        covariate_cols=covariates,
        outcome_col=outcome,
        treatment_col=treatment,
        k_range=range(2, 20, 1),
    )

    # poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    # poly_features = poly.fit_transform(df_dgp[covariates])
    # feature_names = poly.get_feature_names_out(covariates)
    # df_poly = pd.DataFrame(poly_features, columns=feature_names)
    # df_observed = pd.concat([df_poly, df_dgp[[outcome, treatment]]], axis=1)

    # covariates_poly = [
    #     "X1",
    #     "X1^2",
    #     "X2",
    #     "X2^2",
    #     "X1 X2",
    #     # ----------
    #     "X3",
    #     "X3^2",
    #     "X1 X3",
    #     "X2 X3",
    #     "X4",
    #     "X4^2",
    #     "X1 X4",
    #     "X2 X4",
    #     "X3 X4",
    #     # ----------
    #     "X1^3",
    #     "X1^2 X2",
    #     "X2^3",
    #     "X1 X2^2",
    #     # ----------
    #     "X1^2 X3",
    #     "X1^2 X4",
    #     "X1 X2 X3",
    #     "X1 X2 X4",
    #     "X1 X3^2",
    #     "X1 X3 X4",
    #     "X1 X4^2",
    #     "X2^2 X3",
    #     "X2^2 X4",
    #     "X2 X3^2",
    #     "X2 X3 X4",
    #     "X2 X4^2",
    #     "X3^3",
    #     "X3^2 X4",
    #     "X3 X4^2",
    #     "X4^3",
    # ]

    # covariates_x1_x2 = [
    #     "X1",
    #     "X1^2",
    #     "X2",
    #     "X2^2",
    #     "X1 X2",
    #     # ----------
    #     "X1^3",
    #     "X1^2 X2",
    #     "X2^3",
    #     "X1 X2^2",
    #     # ----------
    #     "X3",
    #     "X4",
    # ]

    # dict_awm_covariates = {
    #     "X1": ["X1"],
    #     "X1 Quad": ["X1", "X1^2"],
    #     "X1 Quad X2": [
    #         "X1",
    #         "X1^2",
    #         "X2",
    #     ],
    #     "X1 X2 Quad": [
    #         "X1",
    #         "X1^2",
    #         "X2",
    #         "X2^2",
    #         "X1 X2",
    #     ],
    #     "X1 X2 3rd": [
    #         "X1",
    #         "X1^2",
    #         "X2",
    #         "X2^2",
    #         "X1 X2",
    #         # ----------
    #         "X1^3",
    #         "X1^2 X2",
    #         "X2^3",
    #         "X1 X2^2",
    #     ],
    # }

    # ans = run_awm_covariate_selectoin(
    #     data=df_observed,
    #     main_covariate_cols=covariates_x1_x2,
    #     dict_emw_covariate_cols=dict_awm_covariates,
    #     outcome_col=outcome,
    #     treatment_col=treatment,
    # )

    plot_policy(ans.final_solution, effect_shape=effect_shape)
