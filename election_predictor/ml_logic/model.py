from abc import ABC, abstractmethod
import numpy as np

# Import params
from election_predictor.params import *

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


# Our blueprint for all models using abstract base class
class BaseModel(ABC):
    """
    Abstract Base Class for all models. Defines the interface that all models must implement.
    """

    @abstractmethod
    def initialize_model(self, *args, **kwargs):
        """
        Initialize the model. This method should be implemented to set up the model architecture.
        """
        pass

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        """
        Compile the model. This method should be implemented to configure the model for training.
        """
        pass

    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> tuple[object, dict]:
        """
        Train the model on the provided data.

        Parameters:
        - X: Input features
        - y: Target values

        Returns:
        - Trained model
        - Training history or metrics
        """
        pass

    @abstractmethod
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> dict:
        """
        Evaluate the model on the provided data.

        Parameters:
        - X: Input features
        - y: Target values

        Returns:
        - Evaluation metrics
        """
        pass

# Handle the XGBoost Regressor model
class XGBoostModel(BaseModel):
    """
    XGBoost model implementation.
    """

    def __init__(self):
        self.model = None

    def initialize_model(self):
        self.model = XGBRegressor()

        print("✅ XGBoost Regressor Model initialized")
        return self.model

    def compile_model(
        self,
        model: XGBRegressor,
        learning_rate: XGBOOST_PARAMS["learning_rate"],
        n_estimators: XGBOOST_PARAMS["n_estimators"],
        max_depth: XGBOOST_PARAMS["max_depth"],
        subsample: XGBOOST_PARAMS["subsample"],
        objective: XGBOOST_PARAMS["objective"],
        nthread: XGBOOST_PARAMS["nthread"],
        enable_categorical: XGBOOST_PARAMS["enable_categorical"]
    ) -> XGBRegressor:
        """
        Compiles the XGBoost Regressor model with the provided parameters. Default
        parameters are set from params.py.

        :param learning_rate: Learning rate for the model.
        :param n_estimators: Number of estimators (trees) to build.
        :param max_depth: Maximum depth of the trees.
        :param subsample: Fraction of samples to use for training each tree.
        :param objective: Objective function to optimize.
        :param nthread: Number of threads to use.
        :param enable_categorical: Whether to enable categorical features.
        :return: The trained XGBoost Regressor model.
        """
        print("✅ XGBoost Regressor model compiled")
        return model

    def train_model(
        self,
        model: XGBRegressor,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[XGBRegressor, dict]:
        """
        Trains the XGBoost Regressor model on the provided model.
        :param X: Input features.
        :param y: Target values.
        :return: The trained XGBoost Regressor model.
        """
        print("\nTraining XGBoost Regressor model...")

        model.fit(X, y)

        print("✅ XGBoost Regressor Model trained")

        return model

    def evaluate_model(
        self,
        model: XGBRegressor,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates the XGBoost Regressor model.

        :param model: The trained XGBoost Regressor model.
        :param X: Input features.
        :param y: Target values.
        :return: RMSE scoring in an ndarray.
        """
        print("Evaluating XGBoost Regressor model...")
        # Handle cross validation scoring
        rmse_score = cross_val_score(
            model, X, y, scoring="neg_root_mean_squared_error"
        ).mean()

        print(f"✅ Model evaluated, RMSE score: {rmse_score}")

        return rmse_score
