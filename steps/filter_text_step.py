import pandas as pd
from src.filter_text import (
    FilterTextHandler,
    FixDropTextStrategy
)
from zenml import step

@step
def filter_text_step(
    df: pd.DataFrame,
    length: int = 25
) -> pd.DataFrame:
    """Performs text filtering using FixDropTextStrategy."""
    
    strategy = FilterTextHandler(strategy=FixDropTextStrategy(length))
    df_cleaned = strategy.handle_text_filtering(df)

    return df_cleaned