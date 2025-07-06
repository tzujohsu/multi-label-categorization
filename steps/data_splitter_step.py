from typing import Tuple

import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""

    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df[['content']], df[df.columns[1:]])
    return X_train, X_test, y_train, y_test
