import pandas as pd


class AdaptiveWelfareMaximization:
    def __init__(
        self,
        data: pd.DataFrame,
        covariate_cols: list[str],
        outcome_col: str,
        treatment_col: str,
        propensity_col: str | None = None,
    ):
        """
        Initialize the AdaptiveWelfareMaximization class.

        :param data: DataFrame containing the dataset.
        :param covariate_cols: List of column names for covariates.
        :param outcome_col: Name of the outcome column.
        :param treatment_col: Name of the treatment column.
        """

        indexed_data = (
            data.copy()
            .reset_index(drop=True)
            .reset_index(drop=False)
            .rename(columns={"index": "OID"})
        )

        self.indexed_original_data = indexed_data
        self.data = indexed_data[
            ["OID"]
            + covariate_cols
            + [outcome_col, treatment_col]
            + (([propensity_col] if propensity_col else []))
        ]

        self.covariate_cols = covariate_cols
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.propensity_col = propensity_col
        self.has_propensity = propensity_col is not None

    def cv_sample_splitting(
        self,
        cross_vailidation_folds: int = 5,
        seed: int | None = None,
    ):
        """
        Perform sample splitting for cross-validation and cross-fitting.
        Each row in the DataFrame is assigned a "CFID" (Cross-Fitting ID) and a "CVID" (Cross-Validation ID).
        The data is splitted into `cross_vailidation_folds` folds and for each fold,
        the complement folds are further split into `cross_fitting_folds` folds.
        """
        self.n_cv_folds = cross_vailidation_folds

        # randomly assign "CFID" to each row in the DataFrame
        if seed is not None:
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(
                drop=True
            )
        else:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.data["CVID"] = (self.data.index % self.n_cv_folds).astype(int)
        self.data = self.data.sort_values(by="OID").reset_index(drop=True)

    def _get_cv_data(self, cv_id: int, type: str) -> pd.DataFrame:
        """
        Get a deep copy for a specific cross-validation fold.

        :param cv_id: The ID of the cross-validation fold.
        :return: DataFrame containing the data for the specified fold.
        """
        if cv_id < 0 or cv_id >= self.n_cv_folds:
            raise ValueError(f"cv_id must be between 0 and {self.n_cv_folds - 1}")
        if type == "train":
            return (
                self.data[self.data["CVID"] != cv_id]
                .copy(deep=True)
                .reset_index(drop=True)
            )
        elif type == "test":
            return (
                self.data[self.data["CVID"] == cv_id]
                .copy(deep=True)
                .reset_index(drop=True)
            )
        else:
            raise ValueError("type must be either 'train' or 'test'")

    def _cf_sample_splitting(
        self,
        cross_validation_id: int,
        cross_fitting_folds: int = 5,
        seed: int | None = None,
    ):
        """
        Perform cross-fitting splitting for cross-fitting.
        Each row in the DataFrame is assigned a "CFID" (Cross-Fitting ID).
        The data is splitted into `cross_fitting_folds` folds.
        """
        if cross_validation_id is None:
            raise ValueError("cross_validation_id must be specified for cross-fitting")

        train_data = self._get_cv_data(cross_validation_id, "train")

        if seed is not None:
            train_data = train_data.sample(frac=1, random_state=seed).reset_index(
                drop=True
            )
        else:
            train_data = train_data.sample(frac=1).reset_index(drop=True)

        train_data["CFID"] = (train_data.index % cross_fitting_folds).astype(int)
        train_data = train_data.sort_values(by="OID").reset_index(drop=True)

        return train_data

    def create_cross_fitting_data(
        self,
        cross_fitting_folds: int = 5,
        seed: int | None = None,
    ) -> None:
        """
        Create a DataFrame for cross-fitting with "CFID" assigned to each row.

        :param cross_validation_id: The ID of the cross-validation fold.
        :param cross_fitting_folds: Number of folds for cross-fitting.
        :param seed: Random seed for reproducibility.
        :return: DataFrame with "CFID" assigned.
        """
        if not hasattr(self, "n_cv_folds"):
            raise ValueError("Cross-validation must be performed before cross-fitting")

        self.n_cf_folds = cross_fitting_folds

        self.dict_train_data = {}
        for cv_id in range(self.n_cv_folds):
            self.dict_train_data[cv_id] = self._cf_sample_splitting(
                cross_validation_id=cv_id,
                cross_fitting_folds=cross_fitting_folds,
                seed=seed,
            )

    def estimate_nussance_parameter(
        self,
        cross_validation_id: int,
        p_score_model_factory,
        conditional_mean_model_factory,
    ) -> None:

        train_data = self.dict_train_data[cross_validation_id]

        for cf_id in range(self.n_cf_folds):

            estimation_data = train_data[train_data["CFID"] != cf_id].copy(deep=True)

            print(
                f"Estimating nuisance parameters for CVID {cross_validation_id}, CFID {cf_id}"
            )

            # Fit propensity score model
            if not self.has_propensity:
                p_score_model = p_score_model_factory.create_model()
                p_score_model.fit(
                    estimation_data[self.covariate_cols],
                    estimation_data[self.treatment_col],
                )
                train_data.loc[train_data["CFID"] == cf_id, "p_score"] = (
                    p_score_model.predict_proba(
                        train_data.loc[train_data["CFID"] == cf_id, self.covariate_cols]
                    )[:, 1]
                )

            # Fit conditional mean model
            treated_data = estimation_data[estimation_data[self.treatment_col] == 1]
            control_data = estimation_data[estimation_data[self.treatment_col] == 0]

            conditional_mean_model_treat = conditional_mean_model_factory.create_model()
            conditional_mean_model_treat.fit(
                treated_data[self.covariate_cols],
                treated_data[self.outcome_col],
            )
            train_data.loc[train_data["CFID"] == cf_id, "est_Y(1)"] = (
                conditional_mean_model_treat.predict(
                    train_data.loc[train_data["CFID"] == cf_id, self.covariate_cols]
                )
            )

            conditional_mean_model_control = (
                conditional_mean_model_factory.create_model()
            )
            conditional_mean_model_control.fit(
                control_data[self.covariate_cols],
                control_data[self.outcome_col],
            )
            train_data.loc[train_data["CFID"] == cf_id, "est_Y(0)"] = (
                conditional_mean_model_control.predict(
                    train_data.loc[train_data["CFID"] == cf_id, self.covariate_cols]
                )
            )

            train_data = self._get_dr_score(train_data)

    def estimate_all_cf_nussance_parameters(
        self,
        p_score_model_factory,
        conditional_mean_model_factory,
    ) -> None:
        """
        Estimate nuisance parameters for all cross-validation folds.
        :param p_score_model_factory: Factory to create propensity score model.
        :param conditional_mean_model_factory: Factory to create conditional mean model.
        """
        if not hasattr(self, "n_cv_folds"):
            raise ValueError("Cross-validation must be performed before cross-fitting")

        for cv_id in range(self.n_cv_folds):
            self.estimate_nussance_parameter(
                cross_validation_id=cv_id,
                p_score_model_factory=p_score_model_factory,
                conditional_mean_model_factory=conditional_mean_model_factory,
            )

    def empirical_walfare_maximization(
        self,
        cross_validation_id: int,
        empirical_welfare_maximizor,
    ) -> pd.DataFrame:
        """
        Calculate the empirical welfare maximization for a given cross-validation fold.
        :param cross_validation_id: The ID of the cross-validation fold.
        :return: DataFrame with the empirical welfare maximization results.
        """
        if cross_validation_id < 0 or cross_validation_id >= self.n_cv_folds:
            raise ValueError(f"cv_id must be between 0 and {self.n_cv_folds - 1}")

        train_data = self.dict_train_data[cross_validation_id].copy(deep=True)

        if "tau" not in train_data.columns:
            raise ValueError(
                "Nuisance parameters must be estimated before calculating empirical welfare maximization"
            )

        empirical_welfare_maximizor.fit(
            train_data,
            covariate_cols=self.covariate_cols,
            reward_col="tau",
        )

        if not hasattr(self, "dict_fitted_awm"):
            self.dict_fitted_awm = {}

        self.dict_fitted_awm[cross_validation_id] = empirical_welfare_maximizor

        return empirical_welfare_maximizor

    def _get_dr_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate the Doubly Robust (DR) score for each observation in the DataFrame.
        :param df: DataFrame containing the data with necessary columns.
        :return: DataFrame with the DR score added.
        """
        if not all(
            col in df.columns for col in ["p_score", "est_Y(1)", "est_Y(0)", "D"]
        ):
            raise ValueError(
                "DataFrame must contain p_score, est_Y(1), est_Y(0), and D columns"
            )

        df["tau"] = (
            df["est_Y(1)"]
            - df["est_Y(0)"]
            + df["D"] * (df["Y"] - df["est_Y(1)"]) / df["p_score"]
            - (1 - df["D"]) * (df["Y"] - df["est_Y(0)"]) / (1 - df["p_score"])
        )
        return df


class PropensityScoreModelFactory:
    """
    Factory class to create a propensity score model.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type

    def create_model(self):
        """
        Create a propensity score model based on the specified model type.
        :return: A scikit-learn model instance.
        """
        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class ConditionalMeanModelFactory:
    """
    Factory class to create a conditional mean model.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type

    def create_model(self):
        """
        Create a conditional mean model based on the specified model type.
        :return: A scikit-learn model instance.
        """
        if self.model_type == "OLS":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


if __name__ == "__main__":
    from dev.dgp.diagnal import generate_data
    from utils.visualization import scatter

    df = generate_data(n=500, error_variance=0.2)

    covariates = ["X1", "X2", "X3", "X4"]
    outcome = "Y"
    treatment = "D"

    df_observed = df[covariates + [outcome, treatment]].copy()

    awm = AdaptiveWelfareMaximization(
        df,
        covariates,
        outcome,
        treatment,
    )

    awm.cv_sample_splitting(cross_vailidation_folds=5, seed=0)
    awm.create_cross_fitting_data(cross_fitting_folds=5, seed=0)

    p_score_model_factory = PropensityScoreModelFactory(model_type="logistic")
    conditional_mean_model_factory = ConditionalMeanModelFactory(model_type="OLS")

    awm.estimate_all_cf_nussance_parameters(
        p_score_model_factory=p_score_model_factory,
        conditional_mean_model_factory=conditional_mean_model_factory,
    )

    og_data = awm.indexed_original_data.copy()
    train_data = awm.dict_train_data[0].copy(deep=True)
    train_data = train_data.merge(
        og_data[["OID", "p_score", "Y(1)", "Y(0)"]],
        on="OID",
        how="inner",
        suffixes=("", "_true"),
    )

    y_treat = scatter(
        df=train_data,
        x_col="Y(1)",
        y_col="est_Y(1)",
        color_col="D",
        title="Estimated Y(1) vs True Y(1)",
    )

    y_control = scatter(
        df=train_data,
        x_col="Y(0)",
        y_col="est_Y(0)",
        color_col="D",
        title="Estimated Y(0) vs True Y(0)",
    )

    p_score = scatter(
        df=train_data,
        x_col="p_score_true",
        y_col="p_score",
        color_col="D",
        title="Propensity Score vs Treatment",
    )

    from dev.milp.linear_threshold import LinearThreshold

    awm_solver = LinearThreshold(
        beta_min=-10,
        beta_max=10,
        num_features=2,
    )

    awm.empirical_walfare_maximization(
        cross_validation_id=0,
        empirical_welfare_maximizor=awm_solver,
    )

    results = awm_solver.get_results()

    results["beta"]  # Coefficients

    data_plot = awm_solver.data

    solved = scatter(
        df=data_plot,
        x_col="X1",
        y_col="X2",
        color_col="z",
        title="fitted plicy",
    )
