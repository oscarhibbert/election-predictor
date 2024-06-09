import numpy as np
import pandas as pd
from datetime import datetime
# Import all functions inside data.py from ml_logic
from ml_logic.data import fetch_clean_data


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

    #TODO Train, test split data

    #TODO Handle preprocessing

    #TODO Handle modelling

    #TODO Handle model evaluation

    #TODO Handle prediction
