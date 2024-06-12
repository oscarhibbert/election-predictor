import numpy as np
import pandas as pd
from datetime import datetime

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
    ons_economic_data = clean_data_sources


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
    election_date = datetime.strptime(UK_ELECTIONS["2024"], "%Y-%m-%d")
    #TODO Seperate poll window and days from today until election into seperate vars
    cutoff_date = election_date - datetime.timedelta(days=54)

    last_poll_date = polls_results_trends_ons["enddate"].iloc[-1]
    prediction_date = election_date - datetime.timedelta(days=24)

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
    processed_train_data, processed_test_data, \
    preprocessor_pipeline = preprocessor(train_data, test_data)

    #TODO update feature selection via params.py instead of .drop use .select
    X_train = processed_train_data.drop(
        columns=['enddate_year_month', 'Month_y', 'startdate', 'enddate',
                 'pollster', 'Unnamed: 0', 'next_elec_date', 'days_to_elec',
                 'months_to_elec', 'party_in_power_Labour', 'LAB_ACT', 'CON_ACT',
                 'LIB_ACT', 'GRE_ACT', 'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT',
                 'OTH_ACT']
    )

    X_train = preprocessor_pipeline.fit_transform(train_data)

    X_test = processed_test_data.drop(
        columns=['enddate_year_month', 'Month_y', 'startdate', 'enddate',
                 'pollster', 'Unnamed: 0', 'next_elec_date', 'days_to_elec',
                 'months_to_elec', 'party_in_power_Labour', 'LAB_ACT',
                 'CON_ACT', 'LIB_ACT', 'GRE_ACT', 'BRX_ACT', 'SNP_ACT',
                 'UKI_ACT', 'PLC_ACT', 'OTH_ACT']
    )

    X_test = preprocessor_pipeline.transform(X_test)

    # Handle target data
    selected_targets = ['next_elec_date', 'LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT',
                        'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_ACT']

    y_train = processed_train_data[selected_targets]
    y_test = processed_test_data[selected_targets]

    # Handle y_train results cleaning (create a fake election result)
    # (removes actuals results, we are trying to predict)
    y_train.loc[y_train['next_elec_date'] == election_date,
         ['LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT', 'BRX_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_ACT']] = np.nan

    # Fetch mean value of X_test
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

    # Finally drop month_x columns from X_train and X_test
    X_train.drop(columns=['Month_x'], inplace=True)
    X_test.drop(columns=['Month_x'], inplace=True)

    # Handle training and testing UK GE parties vote share

    # Handle matrix conversion for train and test feature data
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    parties = [
        'LAB', 'CON', 'LIB', 'GRE', 'BRX', 'NAT', 'SNP', 'UKI', 'PLC',
        'OTH'
    ]

    train_test_results = { }

    for party_code, party in parties:
        # Set party target train and test
        party_y_train = y_train[party_code]
        party_y_test = y_test[party_code]

        # Handle XGBoost Regressor modelling
        # Handle model fetch and initialisation
        xgb_regressor = XGBoostModel()
        xgb_regressor.initialize_model()

        # Handle model compiling
        xgb_regressor.compile_model() # Uses default model parameters from params.py

        # Handle model training
        trained_model = xgb_regressor.train_model(X_train, party_y_train)

        # Handle model evaluation
        rmse_score = xgb_regressor.evaluate_model(trained_model, X_test, party_y_test).mean()

        train_test_results[party_code] = {
            "rmse_score": rmse_score,
            "trained_model": trained_model
        }

    #Handle prediction (ensure input features are transform via preprocessor instance)
    predict_features = []
    X_predict = np.array(
        preprocessor_pipeline.transform(predict_features)
    )

    election_predictions = { }

    for party_code, train_test_result in train_test_results:
        trained_model = train_test_results

        predicted_vote = trained_model.predict(X_predict)

        election_predictions[party_code] = {
            "predicted_vote": predicted_vote
        }


#     ########## OLD CODE IS BELOW THIS LINE ##########
#     # Handle data source merging
#     # Merge results on polls and some additional cleaning
#     polls_results = national_polls.merge(national_results, on='election_year', how='left')

#     polls_results.rename(
#         columns= {
#             'BRX': 'BRX_FC',
#             'CON': 'CON_FC',
#             'GRE': 'GRE_FC',
#             'LAB': 'LAB_FC',
#             'LIB': 'LIB_FC',
#             'NAT': 'NAT_FC',
#             'OTH': 'OTH_FC',
#             'PLC': 'PLC_FC',
#             'SNP': 'SNP_FC',
#             'UKI': 'UKI_FC',
#             'BRX_ACTUAL_PERCENTAGE': 'BRX_ACT',
#             'CON_ACTUAL_PERCENTAGE': 'CON_ACT',
#             'GRE_ACTUAL_PERCENTAGE': 'GRE_ACT',
#             'LIB_ACTUAL_PERCENTAGE': 'LIB_ACT',
#             'LABOUR_ACTUAL_PERCENTAGE': 'LAB_ACT',
#             'NAT_ACTUAL_PERCENTAGE': 'NAT_ACT',
#             'PLC_ACTUAL_PERCENTAGE': 'PLC_ACT',
#             'SNP_ACTUAL_PERCENTAGE': 'SNP_ACT',
#             'UKI_ACTUAL_PERCENTAGE': 'UKI_ACT'
#         },
#         inplace=True)

#     polls_results.drop(columns=['Country'], inplace=True)
#     polls_results.rename(columns={'pollster_rating': 'rating'}, inplace=True)
#     polls_results.rename(columns={'next_election_date': 'next_elec_date'}, inplace=True)
#     polls_results.rename(columns={'days_until_next_election': 'days_to_elec'}, inplace=True)
#     polls_results.drop(columns='Geography', inplace=True)
#     polls_results.drop(columns='election_year', inplace=True)

#     # Merge trends on polls_results
#     polls_results_trends = pd.merge(
#         polls_results, national_google_trends,how='left',
#         left_on='enddate_year_month',right_on='Month'
#     )

#     # Merge economic data on polls_results_trends
#     polls_results_trends_economic = pd.merge(
#         polls_results_trends,ons_economic_data,
#         how='left',left_on='enddate_year_month',right_on='Month'
#     )

#     # Train, test split data
#     polls_results_trends_economic['next_elec_date'] = polls_results_trends_economic['next_elec_date'].astype("datetime64[ns]")
#     train_data = polls_results_trends_economic[
#         polls_results_trends_economic['next_elec_date'] < datetime.strptime('2019-12-12', '%Y-%m-%d')
#     ]

#     test_data = polls_results_trends_economic[
#         polls_results_trends_economic['next_elec_date'] == datetime.strptime('2019-12-12', '%Y-%m-%d')
#     ]


#     # Handle feature selection
#     # TODO ensure mismatching features from old feature selection match new feature selection from params.py
#     selected_features = DATA_RETRIEVAL["national_polls"]["feature_selection"]
#     # selected_features = ['index','next_elec_date','NAT_ACT', 'BRX_ACT', 'CON_ACT',
#     #    'GRE_ACT', 'LIB_ACT', 'LAB_ACT', 'PLC_ACT', 'SNP_ACT', 'UKI_ACT',
#     #    'OTH_PERCENTAGE','enddate_year_month','Month']

#     # Handle preprocessing
#     processed_train_data, processed_test_data, \
#     preprocessor_pipeline = preprocessor(train_data, test_data)

#     X_train = processed_train_data.drop(columns=selected_features)
#     X_train = preprocessor_pipeline.fit_transform(X_train)

#     X_test = processed_test_data.drop(columns=selected_features)
#     X_test = preprocessor_pipeline.transform(X_test)

#     # Handle matrix converstion for train and test feature data
#     X_train = np.array(X_train)
#     X_test = np.array(X_test)

#     # Handle target data
#     selected_targets = ['LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT', 'BRX_ACT',
#                 'NAT_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_PERCENTAGE']

#     y_train = processed_train_data[selected_targets]

#     y_test = processed_test_data[selected_targets]


#     # Handle training and testing UK GE parties vote share
#     parties = [
#         'LAB', 'CON', 'LIB', 'GRE', 'BRX', 'NAT', 'SNP', 'UKI', 'PLC',
#         'OTH'
#     ]

#     train_test_results = { }

#     for party_code, party in parties:
#         # Set party target train and test
#         party_y_train = y_train[party_code]
#         party_y_test = y_test[party_code]

#         # Handle XGBoost Regressor modelling
#         # Handle model fetch and initialisation
#         xgb_regressor = XGBoostModel()
#         xgb_regressor.initialize_model()

#         # Handle model compiling
#         xgb_regressor.compile_model() # Uses default model parameters from params.py

#         # Handle model training
#         trained_model = xgb_regressor.train_model(X_train, party_y_train)

#         # Handle model evaluation
#         rmse_score = xgb_regressor.evaluate_model(trained_model, X_test, party_y_test).mean()

#         train_test_results[party_code] = {
#             "rmse_score": rmse_score,
#             "trained_model": trained_model
#         }

#     #Handle prediction (ensure input features are transform via preprocessor instance)
#     predict_features = []
#     X_predict = np.array(
#         preprocessor_pipeline.transform(predict_features)
#     )

#     election_predictions = { }

#     for party_code, train_test_result in train_test_results:
#         trained_model = train_test_results

#         predicted_vote = trained_model.predict(X_predict)

#         election_predictions[party_code] = {
#             "predicted_vote": predicted_vote
#         }

# predict_election()
