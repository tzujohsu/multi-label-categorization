import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, hamming_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        pipeline (Pipeline): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Regression Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> dict:
        """
        Evaluates a classifier .

        Parameters:
        pipeline (Pipeline): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.DataFrame): The testing data labels/target.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error.
        """
        logging.info("Predicting using the trained model.")
        X_test = pipeline.steps[0][1].transform(X_test['content'])
        y_pred = pipeline.steps[1][1].predict(X_test)

        logging.info("Calculating evaluation metrics.")
        acc = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average='weighted')
        # recall = recall_score(y_test, y_pred, average='weighted')
        roc = roc_auc_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        metrics = {"accuracy": acc, "roc": roc, 'f1_macro': f1_macro}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (Pipeline): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.DataFrame): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass
