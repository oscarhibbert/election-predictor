# Imports
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline

import pandas as pd

def preprocessor(train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """
    Preprocesses train and test DataFrames.

    :param train_data: DataFrame, training data.
    :param test_data: DataFrame, test data.

    :return: Dictionary, containing the processed training data, processed test data, and the preprocessor pipeline.

    :example:
    >>> preprocessor = preprocessor(train_data, test_data)

    preprocessor["processed_train_data"] # Processed training data
    preprocessor["processed_test_data"] # Processed test data
    preprocessor["preprocessor_pipeline"] # Preprocessor pipeline
    """
    # Handle numerical transformer
    num_columns_selector = ['samplesize', 'days_to_elec', 'poll_length']
    num_transformer = MinMaxScaler()

    # Handle categorical transformer
    cat_columns_selector = ['rating']
    cat_transformer = make_pipeline(OrdinalEncoder(categories = [['F','D-','D','D+','C-','B','B+','A-']]),MinMaxScaler())

    # Build the preprocessing pipeline
    preproc_pipeline = make_column_transformer(
        (num_transformer, num_columns_selector),
        (cat_transformer, cat_columns_selector),
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Fit and transform training data
    train_data_processed = preproc_pipeline.fit_transform(train_data)

    # Transform test data
    test_data_processed = preproc_pipeline.transform(test_data)

    # Return processed data, and pipeline
    return {
        "processed_train_data": train_data_processed,
        "processed_test_data": test_data_processed,
        "preprocessor_pipeline": preproc_pipeline
    }
