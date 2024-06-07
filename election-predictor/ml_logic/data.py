# Imports
import pandas as pd
from  datetime import datetime
from google.cloud import bigquery
from colorama import Fore, Style

from election_predictor.params import *
from election_predictor.ml_logic.utils.election_utils import find_next_election_date

#TODO Create data cleaning functions for each data source
def clean_national_polls(national_polls_dataframe) -> dict:
    """
    Cleans the national polls Dataframe.

    :param national_polls_dataframe: The national polls Dataframe.
    :return: A dictionary containing cleaned polling data in DataFrames for each country.
    """
    np_dataframe = national_polls_dataframe

    # Handle spaces and ampersands in pollster names
    np_dataframe['pollster'] = np_dataframe['pollster'].str.replace(' ', '').str.replace('&', '').str.replace('-', '')

    # Creates unique index for each poll
    df_uuid = np_dataframe.set_index(np_dataframe['enddate'].dt.strftime('%Y-%m-%d').apply(str).str.replace('-', '_') + '_' + np_dataframe['pollster'])

    # Pivots table to create column for each party
    df = df_uuid.pivot_table(values="votingintention", index=[df_uuid.index,\
                                                                        'startdate', 'enddate', 'pollster', 'samplesize', 'countrycode'], columns=['partycode'])
    df.reset_index(level=['startdate', 'enddate', 'pollster', 'samplesize', 'countrycode'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filters to after January 1, 2004
    df = df[df['enddate'] > '2004-01-01']

    # Removes pollsters with fewer than 10 polls
    pollster_counts = df['pollster'].value_counts()
    valid_pollsters = pollster_counts[pollster_counts >= 10].index
    df = df[df['pollster'].isin(valid_pollsters)]

    # Adds rating column
    df['pollster_rating'] = df['pollster'].map(POLLSTER_RATINGS)

    # Set election dates
    election_dates = [
        datetime.strptime(UK_ELECTIONS["2005"]["date"], "%Y-%m-%d"),
        datetime.strptime(UK_ELECTIONS["2010"]["date"], "%Y-%m-%d"),
        datetime.strptime(UK_ELECTIONS["2015"]["date"], "%Y-%m-%d"),
        datetime.strptime(UK_ELECTIONS["2017"]["date"], "%Y-%m-%d"),
        datetime.strptime(UK_ELECTIONS["2019"]["date"], "%Y-%m-%d"),
        datetime.strptime(UK_ELECTIONS["2024"]["date"], "%Y-%m-%d")
    ]

    # Creates next election date column
    df['next_election_date'] = df['startdate'].apply(lambda x: find_next_election_date(x, election_dates))
    df['days_until_next_election'] = (df['next_election_date'] - df['startdate']).dt.days

    # Creates subsets of polls
    gb_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["great_britain"])] # 4622 polls
    sco_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["scotland"])] # 213 polls
    wal_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["wales"])] #91 polls
    ukm_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["united_kingdom"])] #428 polls

    # Disregard polls:
    nir_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["northern_ireland"])] # Inclined to disregard, only 1 polls
    enw_polls = df[df['countrycode'].isin(UK_ELECTION_COUNTRY_CODES["england_wales"])] # Inclined to disregard, only 1 [poll]

    # Defines relevant parties for each country
    gb_columns = ['startdate', 'enddate', 'pollster', 'samplesize', 'pollster_rating', 'next_election_date', 'days_until_next_election', 'BRX', 'CON', 'GRE', 'LAB', 'LIB', 'NAT', 'OTH', 'PLC', 'SNP', 'UKI']
    sco_columns = ['startdate', 'enddate', 'pollster', 'samplesize', 'pollster_rating', 'next_election_date', 'days_until_next_election', 'BRX', 'CON', 'GRE', 'LAB', 'LIB', 'OTH', 'SNP', 'OTH', 'UKI']
    ukm_columns = ['startdate', 'enddate', 'pollster', 'samplesize', 'pollster_rating', 'next_election_date', 'days_until_next_election', 'BRX', 'CON', 'GRE', 'LAB', 'LIB', 'NAT', 'PLC', 'SNP', 'UKI']
    wal_columns = ['startdate', 'enddate', 'pollster', 'samplesize', 'pollster_rating', 'next_election_date', 'days_until_next_election', 'BRX', 'CON', 'GRE', 'LAB', 'LIB', 'OTH', 'PLC', 'UKI']

    # Filtered dataframe for GB polls with GB parties
    gb_df = gb_polls[gb_columns]
    # Filtered dataframe for Scotland polls with relevant parties
    scotland_df = sco_polls[sco_columns]
    # Filtered dataframe for Wales polls with relevant parties
    wales_df = wal_polls[wal_columns]
    # Filtered dataframe for UK polls with UK parties
    uk_df = ukm_polls[ukm_columns]

    # Return a DataFrame containing polls for each country
    return {
        "great_britain": {
            "polls_dataframe": gb_df,
            "election_code": "GBR"
        },
        "scotland": {
            "polls_dataframe": scotland_df,
            "election_code": "SCO"
        },
        "wales": {
            "polls_dataframe": wales_df,
            "election_code": "WAL"
        },
        "united_kingdom": {
            "polls_dataframe": uk_df,
            "election_code": "UKM"
        }
    }

# Not built as no requirement for continued results cleaning
def clean_national_results(national_results_dataframe) -> dict:
    pass

# Not built as no requirement for continued results cleaning
def clean_constituency_results():
    pass

def clean_national_google_trends(google_trends_dataframe) -> pd.DataFrame:
    """
    Cleans Google Trends Dataframe.

    :param good_trends_dataframe: The Google Trends DataFrame.
    :return: A DataFrame containing the cleaned Google Trends data.
    """
    # Ensure DataFrame has only one Labour Party column and months are in datetime format

    # Set DataFrame
    df = google_trends_dataframe

    # Handle <1 values
    df.replace('<1', 0.5,inplace=True)

    # Handle datatypes
    df = df.astype(
        {'Green Party: (United Kingdom)':'int','Reform UK: (United Kingdom)':'int',\
        'Plaid Cymru: (United Kingdom)':'int','Scottish National Party: (United Kingdom)':'int',\
            'UK Independence Party: (United Kingdom)':'int','British National Party: (United Kingdom)':'int'})

    # Handle column naming conventions
    df.rename(columns={
        'Labour Party: (United Kingdom)': 'LAB_trends',
        'Conservative Party: (United Kingdom)': 'CON_trends',
        'Liberal Democrats: (United Kingdom)': 'LIB_trends',
        'Green Party: (United Kingdom)': 'GRE_trends',
        'Reform UK: (United Kingdom)': 'BRX_trends',
        'Plaid Cymru: (United Kingdom)': 'PLC_trends',
        'Scottish National Party: (United Kingdom)': 'SNP_trends',
        'UK Independence Party: (United Kingdom)': 'UKI_trends',
        'British National Party: (United Kingdom)': 'NAT_trends'},
        inplace=True
    )

    return df

def clean_national_wikipedia(wikipedia_dataframe) -> pd.DataFrame:
    """
    Cleans Wikipedia data.

    :param wikipedia_dataframe: The Google Trends DataFrame.
    :return: A DataFrame containing the cleaned Wikipedia data.
    """
    # Handle datatime formatting
    wikipedia_dataframe['date'] = wikipedia_dataframe.to_datetime(wikipedia_dataframe['date'], format='%Y%m%d')

    return wikipedia_dataframe

#TODO Create clean function for Reddit data once built by CK
def clean_national_reddit():
    pass

#TODO Get data should cache data locally to prevent repeated data loading
def get_data(gcp_project:str,query:str,) -> pd.DataFrame:
    """
    Load data from Google BigQuery and return as a dataframe.
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
