from abc import ABC, abstractmethod
import numpy as np

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

