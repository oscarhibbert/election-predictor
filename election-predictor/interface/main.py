import numpy as np
import pandas as pd
from datetime import datetime

# Import all functions inside data.py from ml_logic
from ml_logic.data import fetch_clean_data

# Import preprocessor
from ml_logic.preprocessor import preprocessor

# Import modelling functions from ml_logic
from ml_logic.model import XGBoostModel

def predict_election(election_year:int) -> dict:
    """
    Predicts the outcome of the specified UK general election.

    :param election_year: The election cycle year to predict.
    :return: A dictionary containing the predicted election results.
    """

    #TODO Handle election cycle date logic
    # Determine the last election cycle


    # Handle data source fetching and cleaning
    data_sources = list(DATA_RETRIEVAL.keys())

    clean_data_sources = fetch_clean_data(data_sources)

    national_polls, national_results, constituency_results, \
    national_google_trends, national_wikipedia, national_reddit, \
    ons_economic_data = clean_data_sources

    # Handle data source merging
    # Merge results on polls and some additional cleaning
    polls_results = national_polls.merge(national_results, on='election_year', how='left')

    polls_results.rename(
        columns= {
            'BRX': 'BRX_FC',
            'CON': 'CON_FC',
            'GRE': 'GRE_FC',
            'LAB': 'LAB_FC',
            'LIB': 'LIB_FC',
            'NAT': 'NAT_FC',
            'OTH': 'OTH_FC',
            'PLC': 'PLC_FC',
            'SNP': 'SNP_FC',
            'UKI': 'UKI_FC',
            'BRX_ACTUAL_PERCENTAGE': 'BRX_ACT',
            'CON_ACTUAL_PERCENTAGE': 'CON_ACT',
            'GRE_ACTUAL_PERCENTAGE': 'GRE_ACT',
            'LIB_ACTUAL_PERCENTAGE': 'LIB_ACT',
            'LABOUR_ACTUAL_PERCENTAGE': 'LAB_ACT',
            'NAT_ACTUAL_PERCENTAGE': 'NAT_ACT',
            'PLC_ACTUAL_PERCENTAGE': 'PLC_ACT',
            'SNP_ACTUAL_PERCENTAGE': 'SNP_ACT',
            'UKI_ACTUAL_PERCENTAGE': 'UKI_ACT'
        },
        inplace=True)

    polls_results.drop(columns=['Country'], inplace=True)
    polls_results.rename(columns={'pollster_rating': 'rating'}, inplace=True)
    polls_results.rename(columns={'next_election_date': 'next_elec_date'}, inplace=True)
    polls_results.rename(columns={'days_until_next_election': 'days_to_elec'}, inplace=True)
    polls_results.drop(columns='Geography', inplace=True)
    polls_results.drop(columns='election_year', inplace=True)

    # Merge trends on polls_results
    polls_results_trends = pd.merge(
        polls_results, national_google_trends,how='left',
        left_on='enddate_year_month',right_on='Month'
    )

    # Merge economic data on polls_results_trends
    polls_results_trends_economic = pd.merge(
        polls_results_trends,ons_economic_data,
        how='left',left_on='enddate_year_month',right_on='Month'
    )

    # Train, test split data
    polls_results_trends_economic['next_elec_date'] = polls_results_trends_economic['next_elec_date'].astype("datetime64[ns]")
    train_data = polls_results_trends_economic[
        polls_results_trends_economic['next_elec_date'] < datetime.strptime('2019-12-12', '%Y-%m-%d')
    ]

    test_data = polls_results_trends_economic[
        polls_results_trends_economic['next_elec_date'] == datetime.strptime('2019-12-12', '%Y-%m-%d')
    ]


    # Handle feature selection
    # TODO ensure mismatching features from old feature selection match new feature selection from params.py
    selected_features = DATA_RETRIEVAL["national_polls"]["feature_selection"]
    # selected_features = ['index','next_elec_date','NAT_ACT', 'BRX_ACT', 'CON_ACT',
    #    'GRE_ACT', 'LIB_ACT', 'LAB_ACT', 'PLC_ACT', 'SNP_ACT', 'UKI_ACT',
    #    'OTH_PERCENTAGE','enddate_year_month','Month']

    # Handle preprocessing
    processed_train_data, processed_test_data, \
    preprocessor_pipeline = preprocessor(train_data, test_data)

    X_train = processed_train_data.drop(columns=selected_features)
    X_train = preprocessor_pipeline.fit_transform(X_train)

    X_test = processed_test_data.drop(columns=selected_features)
    X_test = preprocessor_pipeline.transform(X_test)

    # Handle matrix converstion for train and test feature data
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Handle target data
    selected_targets = ['LAB_ACT', 'CON_ACT', 'LIB_ACT', 'GRE_ACT', 'BRX_ACT',
                'NAT_ACT', 'SNP_ACT', 'UKI_ACT', 'PLC_ACT', 'OTH_PERCENTAGE']

    y_train = processed_train_data[selected_targets]

    y_test = processed_test_data[selected_targets]


    # Handle training and testing UK GE parties vote share
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

    #TODO Handle prediction (ensure input features are transform via preprocessor instance)
    predictions = { }

    for party_code, party in train_test_results:
        pass
