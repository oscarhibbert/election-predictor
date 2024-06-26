import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from colorama import Fore, Style
import requests

#TODO Add support for different types of API requests (e.g., POST, PUT, DELETE)
def api_utility(url: str, params: dict) -> dict:
    """
    Fetch data via an API GET request and return it as a JSON object.

    :param url: The URL of the API GET request endpoint.
    :param params: The parameters to pass to the API GET request.

    :return: A JSON object containing the fetched data.

    :example:
    >>> api_utility(
            "https://api.example.com/data",
            {"param1": "value1", "param2": "value2"}
        )
    """
    try:
        print(Fore.BLUE + "\nLoad data from API..." + Style.RESET_ALL)

        response = requests.get(url, params=params)

        # Handle HTTPError for bad responses (4XX or 5XX)
        response.raise_for_status()

        data = response.json()

        print(f"✅ Data loaded")

        return data

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")  # Handle specific HTTP errors e.g., response code 404 or 500
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")  # Handle errors like DNS failure, refused connection, etc
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")  # Handle timeouts like a slow server response
    except requests.exceptions.RequestException as err:
        print(f"OOps: Something Else: {err}")  # Catch-all for any error related to your request

#TODO Get data should cache data locally to prevent repeated data loading
#TODO Customise print messaging with clearer data info
def gcp_bq_utility(
    gcp_service_account_key:str, gcp_project:str, query:str) -> pd.DataFrame:
        """
        Load data from Google BigQuery and return as a dataframe.

        :param gcp_service_account_key: The filepath to the JSON Google Cloud\
            Platform service account key.
        :param gcp_project: The Google Cloud Platform project ID.
        :param query: The SQL query to run on the BigQuery server.

        :return: A DataFrame containing the fetched BigQuery data.

        :example:
        >>> gcp_bq_utility(
                GCP_SERVICE_ACCOUNT_KEY,
                "my-gcp-project",
                "SELECT * FROM my_table"
            )
        """

        print(Fore.BLUE + "\nLoad data from GCP BigQuery..." + Style.RESET_ALL)

        client = bigquery.Client(
            project=gcp_project,
            credentials=service_account.Credentials.from_service_account_file(
                gcp_service_account_key
            )
        )
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        print(f"✅ Data loaded, with shape {df.shape}")

        return df
