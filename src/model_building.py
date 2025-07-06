import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline, make_pipeline


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, vectorizer: str, params: dict) -> ClassifierMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels/target.

        Returns:
        ClassifierMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Logistic Regression using scikit-learn
class LogisticRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, vectorizer: str, params: dict) -> Pipeline:
        """
        Builds and trains a logistic regression model with MultiOutputClassifier using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas DataFrame.")
        if vectorizer not in ["count", "tfidf"]:
            raise ValueError("Vectorizer must be either 'count' or 'tfidf'.")

        logging.info("Initializing Logistic Regression model")

        pipeline = Pipeline(
            [
                (vectorizer, 
                    TfidfVectorizer() if vectorizer == 'tfidf' else CountVectorizer()),  # Feature scaling
                ("model", MultiOutputClassifier(LogisticRegression(**params))),  # Logistic regression model
            ]
        )

        logging.info("Pipeline built.")
        return pipeline

# Concrete Strategy for LightGBM Classifier
class LightGBMStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, vectorizer: str, params: dict) -> Pipeline:
        """
        Builds and trains a LightGBM model with MultiOutputClassifier.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Lightgbm model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas DataFrame.")
        if vectorizer not in ["count", "tfidf"]:
            raise ValueError("Vectorizer must be either 'count' or 'tfidf'.")

        logging.info("Initializing LightGBM model")

        
        vec = TfidfVectorizer() if vectorizer == "tfidf" else CountVectorizer()
        pipeline = Pipeline([
            (vectorizer, vec),
            (f'clf', MultiOutputClassifier(LGBMClassifier(**params))),
        ])
        
        logging.info("Pipeline built.")
        return pipeline


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, vectorizer: str, params: dict) -> ClassifierMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels/target.

        Returns:
        ClassifierMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train, vectorizer, params)


# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    json_file_path = '../extracted_data/news-categorization.json'
    df = pd.read_json(json_file_path,  orient='records', lines=True)
    df.dropna(subset=['content'], inplace=True)
    X_train = df['content']
    y_train = df['category_level_1']

    # Example usage of Linear Regression Strategy
    model_builder = ModelBuilder(LightGBMStrategy())
    trained_model = model_builder.build_model(X_train, y_train, 'tfidf')
    
    # print(trained_model.steps[-1][1].coef_)  # Print model coefficients, for Logistic Regression only

    pass
