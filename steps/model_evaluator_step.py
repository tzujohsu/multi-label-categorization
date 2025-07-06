import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> dict:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_pipeline (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.DataFrame): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError("y_test must be a pandas DataFrame.")

    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(
        trained_pipeline, X_test, y_test
    )

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")


    return evaluation_metrics
