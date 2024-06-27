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

        self._data_source = dict | pd.DataFrame

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
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_polls_results_combined"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
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
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_polls"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle national election results data
class NationalResults(DataHandler):
    """
    Fetch and clean national results data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_results"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle constituency results data
class ConstituencyResults(DataHandler):
    """
    Fetch and clean constituency results data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["constituency_results"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle constituency bias data
class ConstituencyBias(DataHandler):
    """
    Fetch and clean constituency bias data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["constituency_bias"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle national Google Trends data
class NationalGoogleTrends(DataHandler):
    """
    Fetch and clean Google Trends data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_google_trends"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle national Wikipedia data
class NationalWikipedia(DataHandler):
    """
    Fetch and clean Wikipedia data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_wikipedia"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle ONS economic data
class ONSEconomic(DataHandler):
    """
    Fetch and clean ONS economic data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["ons_economic_data"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Handle national Reddit data
class NationalReddit(DataHandler):
    """
    Fetch and clean Reddit data.
    """
    def __init__(
        self,
        gcp_service_account_key: str,
        gcp_project_id: str,
        start_date: datetime,
        end_date: datetime
    ):
        super().__init__(
            gcp_service_account_key, gcp_project_id, start_date, end_date
        )

    def get_data_source(self):
        self._data_source = gcp_bq_utility(
            self.gcp_service_account_key,
            self.gcp_project_id,
            DATA_RETRIEVAL["national_reddit"]["query"]
        )

    def clean_data(self):
        pass

    def fetch_cleaned_data_source(self):
        return self._data_source

# Factory function handles data retrieval
def data_factory(
    data_sources: str | list, start_date: datetime, end_date: datetime,
    gcp_project_id: str, gcp_service_key: str
) -> dict:
        """
        Factory function that returns the specified data source class.

        :param data_source: The data source class or classes to be returned.
        :param start_date: The start date of the data source(s).
        :param end_date: The end date of the data source(s).
        :param gcp_project_id: The Google Cloud Platform project ID.
        :param gcp_service_key: The Google Cloud Platform service account key.
        :return: A dictionary of the specified data source class(es).

        :example:
        >>> data_factory("national_polls")
        >>> data_factory(["national_polls", "national_results"])
        """

        # Handle edge case for a single data source
        if isinstance(data_sources, str):
            data_sources = [data_sources]

        # Map data source class names in a dictionary
        data_source_classes_map = {
            "national_polls_results_combined": NationalPollsResultsCombined,
            "national_polls": NationalPolls,
            "national_results": NationalResults,
            "constituency_results": ConstituencyResults,
            "constituency_bias": ConstituencyBias,
            "national_google_trends": NationalGoogleTrends,
            "national_wikipedia": NationalWikipedia,
            "ons_economic_data": ONSEconomic,
            "national_reddit": NationalReddit
        }

        gcp_project_id = gcp_project_id
        gcp_service_account_key = gcp_service_key

        data_source_classes = { }

        # Handle instantiation of data source specified classes only
        for data_source in data_sources:
            if data_source in data_source_classes_map:
                data_source_classes[data_source] = data_source_classes_map[data_source](
                    gcp_service_account_key,
                    gcp_project_id,
                    start_date,
                    end_date
                )

            else:
                raise ValueError(
                    f"The data source '{data_source}' does not exist. "
                    "Check the parameter and try again."
                )

        return data_source_classes
