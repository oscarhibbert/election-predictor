# Imports
from abc import ABC, abstractmethod
import pandas as pd
from  datetime import datetime
from google.cloud import bigquery
from colorama import Fore, Style

from election_predictor.params import *
from election_predictor.ml_logic.utils.data_utils import get_data
from election_predictor.ml_logic.utils.election_utils import find_next_election_date
from google.oauth2 import service_account

# Blueprint for all data functions using abstract base class
class DataHandler(ABC):
    """
    Abstract Base Class for all data functions. Defines the interface that all data functions must implement.
    """

    @abstractmethod
    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Initialize the data handler with the specified start and end dates.
        """
        self.start_date = start_date
        self.end_date = end_date
        self._data_source = None

    @abstractmethod
    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the specified source and return it as a DataFrame.

        :return: A DataFrame containing the fetched and cleaned data.

        :example:
        >>> get_data()
        """
        pass

    @abstractmethod
    def clean_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Clean the data and return it as a DataFrame.

        :return: A DataFrame containing the cleaned data.

        :example:
        >>> clean_data()
        """
        pass

# Handle national polls data
class NationalPolls(BaseData):
    """
    Fetch and clean national polls data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_polls"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle national election results data
class NationalResults(BaseData):
    """
    Fetch and clean national election results data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_results"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle national polls and results combined data
class NationalPollsResultsCombined(BaseData):
    """
    Fetch and clean national polls and results combined data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_polls_results_combined"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle constituency results data
class ConstituencyResults(BaseData):
    """
    Fetch and clean constituency results data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["constituency_results"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle constituency bias data
class ConstituencyBias(BaseData):
    """
    Fetch and clean constituency bias data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["constituency_bias"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle national Google Trends data
class NationalGoogleTrends(BaseData):
    """
    Fetch and clean national Google Trends data.
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_google_trends"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle national Wikipedia data
class NationalWikipedia(BaseData):
    """
    Fetch and clean Wikipedia data
    """
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_wikipedia"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle ONS economic data
class ONSEconomic(BaseData):
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["ons_economic_data"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

# Handle national Reddit data
class NationalReddit(BaseData):
    def __init__(self):
        self.gcp_project_id = GCP_PROJECT_ID
        self.data_source = DATA_RETRIEVAL["national_reddit"]["query"]

    def fetch_data(self):
        return get_data(self.gcp_project_id, self.data_source)

    def clean_data(self):
        pass

#TODO Complete factory function
# Factory function handles data retrieval
def data_factory(data_source: str | list, date_range: str) -> BaseData:
    """
    Factory function that returns the specified data source class.

    :param data_source: The data source class or classes to be returned.
    :type data_source: str | list.
    :return: The specified data source class.

    :example:
    >>> data_factory("national_polls")
    """
    if data_source == "national_polls":
        return NationalPolls()

    if data_source == "national_results":
        return NationalResults()

    if data_source == "national_polls_results_combined":
        return NationalPollsResultsCombined()

    if data_source == "constituency_results":
        return ConstituencyResults()

    if data_source == "constituency_bias":
        return ConstituencyBias()

    if data_source == "national_google_trends":
        return NationalGoogleTrends()

    if data_source == "national_wikipedia":
        return NationalWikipedia()

    if data_source == "ons_economic_data":
        return ONSEconomic()

    if data_source == "national_reddit":
        return NationalReddit()

    raise ValueError(f"A specified data source does not exist. Check parameter and try again.")
