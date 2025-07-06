from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass

# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# This strategy inspects the statistics of body text column.
class SummaryStatisticsTextInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints the summary statsitcs of text length for the 'content' column.
        Prints the column distribution of the ground truth column.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nAverage length of the news:")
        print(df['content'].dropna().apply(lambda x: len(x.split())).mean())
        
        print("\nMin and Max length of the news:")
        print(df['content'].dropna().apply(lambda x: len(x.split())).min(), 
              df['content'].dropna().apply(lambda x: len(x.split())).max())
        
        vocab = Counter(" ".join(df['content'].dropna()).split())
        print("\nVocabulary size:", len(vocab))

        print("\ncharacter count: Avg, Min, Max")
        df['char_count'] = df['content'].dropna().apply(len)
        print(round(df['char_count'].mean(), 3), round(df['char_count'].min(), 3), round(df['char_count'].max(), 3))


# This strategy inspects the text length histogram distribution
class TextLengthHistogramInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects the text length histogram distribution

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
    
        # Calculate the length of the content
        df['content_length'] = df['content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['content_length'], bins=50, edgecolor='k')
        plt.title('Content Length Distribution (by white space split)')
        plt.xlabel('Content Length')
        plt.ylabel('Frequency')
        plt.show()

# This strategy inspects the summary statistics of target categories.
class TargetCategoriesBasicInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints the summary statsitcs of target categories

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        target_cols = df.columns[17:34]
        category_counts = df[target_cols].sum().sort_values(ascending=False)

        category_percentages = (category_counts / len(df)) * 100

        category_stats = pd.DataFrame({'Count': category_counts, 'Percentage': category_percentages})
        print('\nCategory Stats:')
        print(category_stats)

        df['num_labels'] = df[target_cols].sum(axis=1)
        print('\nNumber of labels per news:')
        print(df['num_labels'].describe())


# This strategy inspects the correlation of target categories.
class TargetCategoriesCorrelationInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints the summary statsitcs of target categories

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        target_cols = df.columns[17:34]
        plt.figure(figsize=(12,8))
        sns.heatmap(df[target_cols].corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Between Categories")
        plt.show()





# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


# Example usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass
