import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class FilterTextStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class FixDropTextStrategy(FilterTextStrategy):
    def __init__(self, length = 25):
        """
        Initializes the DropZeroLengthStrategy with specific parameters.

        Parameters:
        length (int): The minimum length of text to be considered.
        
        """
        self.length = length
        

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping news with word count (length) <= {self.length}")
        df['content_length'] = df['content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        df_cleaned = df[df['content_length'] >= self.length]
        logging.info(f"News Filtered by length >= {self.length}.")
        df_cleaned = df_cleaned.drop(columns=['content_length'])
        return df_cleaned


# Context Class for Handling Missing Values
class FilterTextHandler:
    def __init__(self, strategy: FilterTextStrategy):
        """
        TODO

        Parameters:
        strategy (FilterTextStrategy): The strategy to be used for filter text.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FilterTextStrategy):
        """
        Sets a new strategy for the FilterTextHandler.

        Parameters:
        strategy (FilterTextStrategy): The new strategy to be used for filter text.
        """
        logging.info("Switching filtering text handling strategy.")
        self._strategy = strategy

    def handle_text_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing news that are too short in length.

        Returns:
        pd.DataFrame: The DataFrame with news filtered.
        """
        logging.info("Executing text filtering handling strategy.")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # json_file_path = '../extracted_data/news-categorization.json'
    # df = pd.read_json(json_file_path,  orient='records', lines=True)
    # print(df.shape)
    # # Initialize missing value handler with a specific strategy
    # text_filtering_handler = FilterTextHandler(FixDropTextStrategy(length=25))
    # df_cleaned = text_filtering_handler.handle_text_filtering(df)

    # print(df_cleaned.shape)

    pass
