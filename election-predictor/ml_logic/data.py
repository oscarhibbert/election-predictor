# Imports
import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style

from election_predictor.params import *

#TODO Create data cleaning functions for each data source
def clean_national_polls():
    pass

def clean_national_results():
    pass

def clean_constituency_results():
    pass

def clean_national_google_trends():
    pass

def clean_national_wikipedia():
    pass

def clean_national_reddit():
    pass

#TODO Get data should cache data locally to prevent repeated data loading
def get_data(gcp_project:str,query:str,) -> pd.DataFrame:
    """
    Load data from Google BigQuery and return as a dataframe
    """
    print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

# Create master fetch and clean function
def fetch_clean_data(data_source) -> list:
    """
    Selects the specified data sources, cleans and returns them as DataFrames.

    :param data_source: The data sources to be fetched and cleaned.
    :type data_source: str or list
    :return: A list of cleaned DataFrames.

    :example:
    >>> fetch_clean_data("national_polls")

    OR

    >>> fetch_clean_data(["national_polls", "national_results"])
    """
    # Set data sources from params
    data_sources = DATA_SOURCES

    # Set list for return
    data = []

    # Handle empty parameter
    if data_source is None:
        raise ValueError("No data source specified. Please specify a data source.")

    # Handle invalid parameter dtype. Must be either a string or a list
    if not isinstance(data_source, (str, list)):
        raise ValueError("Invalid parameter type. Please specify a str or list.")

    # Handle invalid data source
    if data_source not in data_sources:
        raise ValueError("Invalid data source. Please specify a valid data source.")

    # Handle national polls
    if data_source == "national_polls":
        national_polls = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["national_polls"]["query"]
        )

        national_polls_cleaned = clean_national_polls(national_polls)

        data.append(national_polls_cleaned)

    if data_source == "national_results":
        national_results = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["national_results"]["query"]
        )

        national_results_cleaned = clean_national_results(national_results)

        data.append(national_results_cleaned)

    if data_source == "constituency_results":
        constituency_results = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["constituency_results"]["query"]
        )

        constituency_results_cleaned = clean_constituency_results(constituency_results)

        data.append(constituency_results_cleaned)

    if data_source == "national_google_trends":
        national_google_trends = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["national_google_trends"]["query"]
        )

        national_google_trends_cleaned = clean_national_google_trends(national_google_trends)

        data.append(national_google_trends_cleaned)

    if data_source == "national_wikipedia":
        national_wikipedia = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["national_wikipedia"]["query"]
        )

        national_wikipedia_cleaned = clean_national_wikipedia(national_wikipedia)

        data.append(national_wikipedia_cleaned)

    if data_source == "national_reddit":
        national_reddit = get_data(
            GCP_PROJECT_ID,
            DATA_RETRIEVAL["national_reddit"]["query"]
        )

        national_reddit_cleaned = clean_national_reddit(national_reddit)

        data.append(national_reddit_cleaned)

    return data
