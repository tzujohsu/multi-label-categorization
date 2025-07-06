import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import re
import string
import nltk
from nltk.corpus import stopwords

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


class BasicPreprocessText(FeatureEngineeringStrategy):
    def __init__(self, feature):
        """
        Initializes the Basic Transformation with the target feature to transform.

        Parameters:
        feature: The feature to apply the preprocessing to.
        """
        self.feature = feature

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying text preprocessing (basic cleaning) to text:")
        def preprocess_text(text):
            return re.sub('\[.*?\]|\w*\d\w*|https?://\S+|www\.\S+|<.*?>+|[%s]' %
                re.escape(string.punctuation), '', str(text).lower())
        
        df_transformed = df.copy()
        df_transformed[self.feature] = df_transformed[self.feature].apply(preprocess_text)
        logging.info("Basic Preprocessing completed.")
        return df_transformed

class RemoveStopwords(FeatureEngineeringStrategy):
    def __init__(self, feature):
        """
        Initializes the Stopword Removing with the target feature to transform.

        Parameters:
        feature: The feature to apply the stop removing to.
        """
        self.feature = feature

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying stopword removing to text:")
        stop_words = stopwords.words('english')
        def remove_stopword(text):
            return ' '.join(word for word in
                text.split(' ') if word not in stop_words)
        
        df_transformed = df.copy()
        df_transformed[self.feature] = df_transformed[self.feature].apply(remove_stopword)
        logging.info("Stopword Removing completed.")
        return df_transformed
    
class TextStemming(FeatureEngineeringStrategy):
    def __init__(self, feature):
        """
        Initializes the Stemming for the target feature.

        Parameters:
        feature: The feature to apply the stop removing to.
        """
        self.feature = feature

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Stemming to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying Stemming to text:")
        stemmer = nltk.SnowballStemmer("english")
        def apply_stemming(sentence):
            return ' '.join(stemmer.stem(word) for word in sentence.split(' '))
        
        df_transformed = df.copy()
        df_transformed[self.feature] = df_transformed[self.feature].apply(apply_stemming)
        logging.info("Text Stemming completed.")
        return df_transformed

# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    json_file_path = '../extracted_data/news-categorization.json'
    df = pd.read_json(json_file_path,  orient='records', lines=True)
    print(df.shape)
    print(df['content'].head())

    # Basic Preprocessing Example
    basic_preprocess = FeatureEngineer(BasicPreprocessText(feature='content'))
    df = basic_preprocess.apply_feature_engineering(df)
    print(df['content'].head())

    # Stopword Removing Example
    stopword_remover = FeatureEngineer(RemoveStopwords(feature='content'))
    df = stopword_remover.apply_feature_engineering(df)
    print(df['content'].head())

    # Text Stemming Example
    text_stemming = FeatureEngineer(TextStemming(feature='content'))
    df = text_stemming.apply_feature_engineering(df)
    print(df['content'].head())
    
    pass
