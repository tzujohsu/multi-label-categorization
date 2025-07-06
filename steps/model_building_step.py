import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from src.model_building import (
    ModelBuilder,
    LogisticRegressionStrategy,
    LightGBMStrategy
)

from zenml import ArtifactConfig, step
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="multi-label news categorization classifier",
    version=None,
    license="Apache 2.0",
    description="News Categorization Model",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_type: str = "logistic_regression",
    vectorizer_type: str = 'tfidf',
    params: dict = {},
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a classifier using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.DataFrame): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """
    
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError("y_train must be a pandas DataFrame.")
    
    # Initialize the vectorizer
    if vectorizer_type not in ["count", "tfidf"]:
        raise ValueError("Vectorizer must be either 'count' or 'tfidf'.")
    

    # Initialize the model builder
    if model_type == "logistic_regression":
        model_builder = ModelBuilder(LogisticRegressionStrategy())
    elif model_type == "lightgbm":
        model_builder = ModelBuilder(LightGBMStrategy())
    
    # Build the model pipeline
    pipeline = model_builder.build_model(X_train, 
                                         y_train, 
                                         vectorizer_type,
                                         params)
    
    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active
    try:
        if model_type != 'lightgbm':
            mlflow.sklearn.autolog()
            # mlflow.lightgbm.autolog(disable=True)
        logging.info("Building and training the model.")
        
        X_train = pipeline.steps[0][1].fit_transform(X_train['content'])
        pipeline.steps[-1][1].fit(X_train, y_train)

        if model_type == 'lightgbm':
            mlflow.lightgbm.log_model(pipeline.steps[-1][1], 'lightgbm_model')
        logging.info("Model training completed.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline
