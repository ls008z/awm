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
