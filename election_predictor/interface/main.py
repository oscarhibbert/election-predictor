import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import params
from election_predictor.params import *

# Import all functions inside data.py from ml_logic
from election_predictor.ml_logic.data import fetch_clean_data

# Import preprocessor
from election_predictor.ml_logic.preprocessor import preprocessor

# Import modelling functions from ml_logic
from election_predictor.ml_logic.model import XGBoostModel

def predict_election() -> dict:
    """
    Predicts the outcome of the 2024 UK general election.

    :return: A dictionary containing the predicted general election vote share and projected seats.
    """

    # # Handle election cycle date logic
    # election_years = UK_ELECTIONS.keys()

    # if str(election_year) not in election_years:
    #     raise ValueError(f"{election_year} isn't an election year. Please provide a valid election year.")

    # # Handle selecting the last year election year, prior to specified election year
    # election_years_ints = [int(year) for year in election_years]

    # # Filter the years to include only those less than or equal to the given election_year
    # past_election_years = [year for year in election_years_ints if year < election_year]

    # if not past_election_years:
    #     raise ValueError(f"No elections found before the year {election_year}.")

    # last_election_year = max(past_election_years)


    # # Handle data source date start and end range
    # data_source_range_start = DATA_SOURCES_START_DATE
    # data_source_range_end = UK_ELECTIONS[str(last_election_year)]["date"]

    # Handle data source fetching and cleaning
    data_sources = ["national_polls_results_combined","constituency_bias", "national_google_trends","ons_economic_data"]

    clean_data_sources = fetch_clean_data(data_sources)

    national_polls_results_combined, constituency_bias, national_google_trends, \
    ons_economic_data = clean_data_sources.values()

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

    # Handle manual scaling for trends columns
    for column in ['LAB_trends', 'CON_trends', 'LIB_trends',
       'GRE_trends', 'BRX_trends', 'PLC_trends', 'SNP_trends', 'UKI_trends',
       'NAT_trends']:
            polls_results_trends_ons[column] = polls_results_trends_ons[column] / 100

    # Handle election cycle date logic
    election_date = UK_ELECTIONS.get("2024")
    election_date = datetime.strptime(election_date["date"], "%Y-%m-%d")

    #TODO Seperate poll window and days from today until election into seperate vars
    cutoff_date = election_date - timedelta(days=54)

    last_poll_date = polls_results_trends_ons["enddate"].iloc[-1]
    prediction_date = election_date - timedelta(days=24)

    # Handle train and test data splitting
    train_data = polls_results_trends_ons[
        polls_results_trends_ons["startdate"] > "2003-12-31"]

    train_data = train_data[train_data["startdate"] < cutoff_date]

    test_data = polls_results_trends_ons[
        (polls_results_trends_ons[
            "startdate"] >= cutoff_date) & (polls_results_trends_ons[
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


    # Handle training and testing UK GE parties vote share

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
        party_y_test = y_test[party_code + "_ACT"]

        # Handle XGBoost Regressor modelling
        # Handle model fetch and initialisation
        xgb_regressor = XGBoostModel()
        xgbr_model = xgb_regressor.initialize_model()

        # Handle model compiling
        xgb_regressor.compile_model(xgbr_model) # Uses default model parameters from params.py

        # Handle model training
        trained_model = xgb_regressor.train_model(xgbr_model, X_train_matrix, party_y_train)

        trained_models[party_code] = trained_model

        #TODO Reintroduce model evaluation logic for RMSE
        # # Handle model evaluation
        # rmse_score = xgb_regressor.evaluate_model(trained_model, X_test_matrix, party_y_test).mean()

        # train_test_results[party_code] = {
        #     "rmse_score": rmse_score,
        #     "trained_model": trained_model
        # }

    #Handle prediction (ensure input features are transformed via preprocessor instance)
    X_predict_matrix = np.array(X_test)

    election_predictions = { }

    for party_code, model in trained_models.items():
        trained_model = model

        predicted_vote_share = trained_model.predict(X_predict_matrix).mean()

        election_predictions[party_code] = predicted_vote_share

    return election_predictions

predict_election()
