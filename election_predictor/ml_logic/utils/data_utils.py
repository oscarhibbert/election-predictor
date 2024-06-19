import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from colorama import Fore, Style


#TODO Get data should cache data locally to prevent repeated data loading
def get_data(gcp_project:str,query:str,) -> pd.DataFrame:
    """
    Load data from Google BigQuery and return as a dataframe.
    """
    credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_KEY)

    print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)

    client = bigquery.Client(project=gcp_project, credentials=credentials)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    print(f"✅ Data loaded, with shape {df.shape}")

    return df
