# Imports
import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style

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
