import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    BasicPreprocessText,
    RemoveStopwords,
    TextStemming,
)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame,
      basic: bool = True, 
      stopword_removal: bool = True,
      stemming: bool = True,
      features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required
    feature = features[0]
    
    if basic:
        engineer = FeatureEngineer(BasicPreprocessText(feature))
        df = engineer.apply_feature_engineering(df)
    if stopword_removal:
        engineer = FeatureEngineer(RemoveStopwords(feature))
        df = engineer.apply_feature_engineering(df)
    if stemming:
        engineer = FeatureEngineer(TextStemming(feature))
        df = engineer.apply_feature_engineering(df)
    
    relevant_cols = ['content']
    relevant_cols.extend(list(df.columns[17:34]))
    
    df = df[relevant_cols]
    
    return df
