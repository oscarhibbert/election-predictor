# Imports
from typing import Literal
from abc import ABC, abstractmethod
import pandas as pd
from  datetime import datetime
from colorama import Fore, Style

from election_predictor.params import *
from election_predictor.ml_logic.utils.data_utils import gcp_bq_utility, api_utility
from election_predictor.ml_logic.utils.election_utils import find_next_election_date
from google.oauth2 import service_account

# Blueprint for all data functions using abstract base class
class DataHandler(ABC):
    """
    Abstract Base Class for all data functions. Defines the interface that\
    all data functions must implement.
    """

    @abstractmethod
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        data_source_start_date: datetime,
        data_source_end_date: datetime
    ):
        """
        Initialize the data handler with the specified start and end dates.

        :param gcp_service_account_key: The filepath to the JSON Google Cloud \
            account key.
        :param gcp_project_id: The Google Cloud Platform project ID.
        :param data_source_start_date: The start date of the data source.
        :param data_source_end_date: The end date of the data source.
        """
        self.gcp_service_account_key = gcp_service_account_key
        self.gcp_project_id = gcp_project_id
        self.data_source_start_date = data_source_start_date
        self.data_source_end_date = data_source_end_date
        self._data_source = None

    @abstractmethod
    def get_data_source(self, *args, **kwargs) -> pd.DataFrame | dict:
        """
        Get data from the specified source and save it as a DataFrame or dictionary\
        to the data source property.

        :example:
        >>> get_data()
        """
        pass

    @abstractmethod
    def clean_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Clean data saved into the data source property and overwrite the result\
        into the data source property.

        :example:
        >>> clean_data()
        """
        pass

    @abstractmethod
    def fetch_cleaned_data_source(self, *args, **kwargs) -> pd.DataFrame:
        """
        Fetch the cleaned data source, returns a DataFrame.

        :return: A DataFrame containing the fetched and cleaned data.

        :example:
        >>> fetch_cleaned_data()
        """
        pass

# Handle national polls and results combined data
class NationalPollsResultsCombined(DataHandler):
    """
    Fetch and clean national polls and results combined data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime,
        _data_source: pd.DataFrame | dict
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_polls_results_combined"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle national polls data
class NationalPolls(DataHandler):
    """
    Fetch and clean national polls data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime,
        _data_source: pd.DataFrame | dict
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_polls"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle national election results data
class NationalResults(DataHandler):
    """
    Fetch and clean national results data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_results"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle constituency results data
class ConstituencyResults(DataHandler):
    """
    Fetch and clean constituency results data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["constituency_results"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle constituency bias data
class ConstituencyBias(DataHandler):
    """
    Fetch and clean constituency bias data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["constituency_bias"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle national Google Trends data
class NationalGoogleTrends(DataHandler):
    """
    Fetch and clean Google Trends data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_google_trends"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source


# Handle national Wikipedia data
class NationalWikipedia(DataHandler):
    """
    Fetch and clean Wikipedia data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_wikipedia"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle ONS economic data
class ONSEconomic(DataHandler):
    """
    Fetch and clean ONS economic data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["ons_economic_data"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

# Handle national Reddit data
class NationalReddit(DataHandler):
    """
    Fetch and clean Reddit data.
    """
    def __init__(
        self,
        gcp_service_account_key,
        gcp_project_id,
        start_date,
        end_date,
        _data_source
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date,
            _data_source
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_reddit"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_clean_data_source(self):
        return self._data_source

#TODO Complete factory function
# Factory function handles data retrieval
def data_factory(
    data_source: str | list, start_date: datetime, end_date: datetime
) -> DataHandler:
        """
        Factory function that returns the specified data source class.

        :param data_source: The data source class or classes to be returned.
        :param start_date: The start date of the data source(s).
        :param end_date: The end date of the data source(s).
        :return: The specified data source class(es).

        :example:
        >>> data_factory("national_polls")
        >>> data_factory(["national_polls", "national_results"])
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
