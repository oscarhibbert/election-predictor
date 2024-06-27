import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from colorama import Fore, Style

# Import params
from election_predictor.params import *

# Import all functions inside data.py from ml_logic
from election_predictor.ml_logic.data import data_factory

# Import preprocessor
from election_predictor.ml_logic.preprocessor import preprocessor

# Import modelling functions from ml_logic
from election_predictor.ml_logic.model import XGBoostModel


def fetch_data() -> dict:
    """
    Fetches data for specified data sources, should be used in the predict
    election function.

    :return: A dictionary containing national polls and results combined,
    constituency bias, national google trends and ONS economic data.
    """
    print(Fore.GREEN + "\nFetching data..." + Style.RESET_ALL)

    # Handle data source fetching and cleaning
    data_sources_start_date = DATA_SOURCES_START_DATE
    data_sources_end_date = DATA_SOURCES_END_DATE
    data_sources = [
        "national_polls_results_combined","constituency_bias",
        "national_google_trends","ons_economic_data"
    ]

    # Fetch specified data source classes from data.py
    data_source_classes = \
        data_factory(data_sources, data_sources_start_date,
            data_sources_end_date, GCP_PROJECT_ID,
            GCP_SERVICE_ACCOUNT_KEY
        )

    # Fetch actual data via data source classes
    clean_data_sources = { }

    for data_source_name, data_source_class in data_source_classes.items():
        data_source_class.get_data_source()
        data_source_class.clean_data()
        clean_data_sources[data_source_name] = \
            data_source_class.fetch_cleaned_data_source()

    print(Fore.GREEN + "\n✅ Data Fetching Complete" + Style.RESET_ALL)

    return clean_data_sources

def clean_data(data_sources: dict) -> pd.DataFrame:
    """
    Cleans the fetched data sources. Should be used in the predict election
    function.

    :param: data_sources: A dictionary containing the data sources from
    fetch_data.

    :return: A DataFrame combining cleaned polls, results,
    trends and ONS data – ready for preprocessing.
    """
    print(Fore.GREEN + "\nCleaning data..." + Style.RESET_ALL)

    national_polls_results_combined, constituency_bias, \
    national_google_trends, ons_economic_data = data_sources.values()

    # Handle polls and results combined cleaning
    national_polls_results_combined['enddate'] = \
        pd.to_datetime(national_polls_results_combined['enddate'])

    national_polls_results_combined['next_elec_date'] = \
        pd.to_datetime(national_polls_results_combined['next_elec_date'])

    national_polls_results_combined['startdate'] = \
        pd.to_datetime(national_polls_results_combined['startdate'])

    # Create poll end date field pre-join with Trends and Economic data
    national_polls_results_combined['enddate_year_month'] = \
        pd.to_datetime(national_polls_results_combined['enddate']).dt.to_period('M')

    national_polls_results_combined.enddate_year_month = \
        pd.to_datetime(national_polls_results_combined.enddate_year_month.astype('str'))

    #TODO Handle dtype clean for the Month column in national_google_trends in data.py
    national_google_trends['Month'] = pd.to_datetime(national_google_trends['Month'])

    # Merge polls_results_combined with national_trends
    polls_results_trends = \
        pd.merge(
            national_polls_results_combined,
            national_google_trends,
            how='left',
            left_on='enddate_year_month',
            right_on='Month'
        )

    # Handle ons_economic_data cleaning
    ons_economic_data['Month'] = pd.to_datetime(ons_economic_data['Month'])

    # Merge polls_results_trends with ons_economic_data
    polls_results_trends_ons = \
        pd.merge(
            polls_results_trends,
            ons_economic_data,
            how='left',
            left_on='enddate_year_month',
            right_on='Month'
        )

    print(Fore.GREEN + "\n✅ Data Cleaning Complete" + Style.RESET_ALL)

    return polls_results_trends_ons

def preprocess_data(cleaned_data: pd.DataFrame) -> dict:
    """
    Preprocesses the combined national polls, results, Google trends and ONS
    data.

    :param: clean_data_sources: A DataFrame containing the the combined
    national polls, results, Google trends and ONS data.

    :return: A dictionary containing X and y train and test data in a dictionary
    ready for model training and evaluation.
    """
    print(Fore.GREEN + "\nPreprocessing data..." + Style.RESET_ALL)

    # Handle manual scaling for trends columns
    for column in ['LAB_trends', 'CON_trends', 'LIB_trends',
       'GRE_trends', 'BRX_trends', 'PLC_trends', 'SNP_trends', 'UKI_trends',
       'NAT_trends']:
            cleaned_data[column] = cleaned_data[column] / 100

    # Handle election cycle date logic
    election_date = UK_ELECTIONS.get("2024")
    election_date = datetime.strptime(election_date["date"], "%Y-%m-%d")

    #TODO Seperate poll window and days from today until election into seperate vars
    cutoff_date = election_date - timedelta(days=45)

    last_poll_date = cleaned_data["enddate"].iloc[-1]
    prediction_date = election_date - timedelta(days=15)

    # Handle train and test data splitting
    train_data = cleaned_data[
        cleaned_data["startdate"] > "2003-12-31"]

    train_data = train_data[train_data["startdate"] < cutoff_date]

    test_data = cleaned_data[
        (cleaned_data[
            "startdate"] >= cutoff_date) & (cleaned_data[
                "startdate"] <= prediction_date)]

    test_data = test_data[test_data["next_elec_date"] == election_date]

    # Handle preprocessing
    preprocessor_pipeline = preprocessor()

    #TODO Update feature selection via params.py instead of .drop use .select
    #TODO Refactor preproc logic into preprocessor.py

    drop_columns = ['enddate_year_month', 'Month_y', 'startdate', 'enddate',
                 'pollster', 'next_elec_date', 'days_to_elec',
                 'months_to_elec', 'LAB_ACT', 'CON_ACT',
                 'LIB_ACT', 'GRE_ACT', 'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT',
                 'OTH_ACT', 'int64_field_0_x', 'int64_field_0_y']


    X_train = train_data.drop(columns=drop_columns)

    #TODO Review NaNs logic with Niek and Chris – refactor into preproc?
    X_train = X_train.fillna(value=0)

    X_train = preprocessor_pipeline.fit_transform(X_train)

    X_test = test_data.drop(columns=drop_columns)

    X_test = X_test.fillna(value=0)

    X_test = preprocessor_pipeline.transform(X_test)

    # Convert train and test features back into DataFrames
    X_train = pd.DataFrame(X_train, columns=preprocessor_pipeline.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=preprocessor_pipeline.get_feature_names_out())

    # Handle target data
    selected_targets = ['next_elec_date', 'LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT',
                        'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_ACT']

    y_train = train_data[selected_targets]
    y_test = test_data[selected_targets]

    # Handle y_train results cleaning (create a fake election result)
    # (removes actuals results, we are trying to predict)
    y_train.loc[y_train['next_elec_date'] == election_date,
         ['LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT', 'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_ACT']] = np.nan

    # Fetch mean values of each value in X_test
    X_test_median = X_test.mean()

    # Create imputation for y_train to impute over actuals we are trying to predict
    imputation_values = {
        'CON_ACT': X_test_median['CON_FC'],
        'LAB_ACT': X_test_median['LAB_FC'],
        'LIB_ACT': X_test_median['LIB_FC'],
        'BRX_ACT': X_test_median['BRX_FC'],
        'GRE_ACT': X_test_median['GRE_FC'],
        'OTH_ACT': X_test_median['OTH_FC'],
        'PLC_ACT': X_test_median['PLC_FC'],
        'SNP_ACT': X_test_median['SNP_FC'],
        'UKI_ACT': X_test_median['UKI_FC']
    }

    y_train = y_train.fillna(value=imputation_values)
    y_train = y_train.fillna(value=0)

    # Finally drop month_x columns from X_train and X_test
    X_train.drop(columns=['Month_x'], inplace=True)
    X_test.drop(columns=['Month_x'], inplace=True)

    print(Fore.GREEN + "\n✅ Data Preprocessing Complete" + Style.RESET_ALL)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """
    Trains the specified models on the training data, for each
    party.

    :param: X_train: The training data features.
    :param: y_train: The training data targets.

    :return: A dictionary containing the trained models for each
    party.
    """
    print(Fore.GREEN + "\nTraining models..." + Style.RESET_ALL)

    # Handle matrix conversion for train feature data
    X_train_matrix = np.array(X_train)

    parties = [
        'LAB', 'CON', 'LIB', 'GRE', 'BRX', 'SNP', 'UKI', 'PLC',
        'OTH'
    ]

    trained_models = { }

    for party, party_code in enumerate(parties):
        # Set party target train and test
        party_y_train = y_train[party_code + "_ACT"]
        # party_y_test = y_test[party_code + "_ACT"]

        # Handle XGBoost Regressor modelling
        # Handle model fetch and initialisation
        xgb_regressor = XGBoostModel()
        xgbr_model = xgb_regressor.initialize_model()

        # Handle model compiling
        # Uses default model parameters from params.py
        xgb_regressor.compile_model(xgbr_model)

        # Handle model training
        trained_model = \
            xgb_regressor.train_model(xgbr_model, X_train_matrix, party_y_train)

        trained_models[party_code] = trained_model

    print(Fore.GREEN + "\n✅ Model Training Complete" + Style.RESET_ALL)

    return trained_models

#TODO Introduce model evaluation scoring and previous election delta
# performance evaluation
def evaluate_models():
    pass
    # #  Handle training and testing UK GE parties vote share

    # # TODO Reintroduce model evaluation logic for RMSE
    # # Handle model evaluation
    # rmse_score = xgb_regressor.evaluate_model(trained_model, X_test_matrix, party_y_test).mean()

    # train_test_results[party_code] = {
    #     "rmse_score": rmse_score,
    #     "trained_model": trained_model
    # }

def predict(trained_models: dict, X_predict: pd.DataFrame) -> dict:
    """
    Predicts the outcome of the 2024 UK general election.

    :param: trained_models: A dictionary containing the trained models for each
    :param: X_predict: The features to predict on.

    :return: A dictionary containing the predicted general election vote
    share and projected seats.
    """
    print(Fore.GREEN + "\nRunning prediction..." + Style.RESET_ALL)

    #Handle prediction (ensure input features are transformed via preprocessor instance)
    X_predict_matrix = np.array(X_predict)

    election_predictions = { }

    for party_code, model in trained_models.items():
        trained_model = model

        predicted_vote_share = trained_model.predict(X_predict_matrix).mean()

        election_predictions[party_code] = predicted_vote_share

    print(Fore.GREEN + "\n✅ Prediction Complete" + Style.RESET_ALL)

    return election_predictions

def predict_election() -> dict:
    """
    Predicts the outcome of the 2024 UK general election.

    :return: A dictionary containing the predicted general election vote
    share and projected seats.
    """

    data_sources = fetch_data()

    cleaned_data = clean_data(data_sources)

    preprocessed_data = preprocess_data(cleaned_data)

    X_train, y_train, X_test, y_test = \
        preprocessed_data['X_train'], \
        preprocessed_data['y_train'], \
        preprocessed_data['X_test'], \
        preprocessed_data['y_test']

    trained_models = train_models(X_train, y_train)

    prediction = predict(trained_models, X_test)

    print(prediction)

    return prediction

predict_election()
