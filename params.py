import os

##################  GOOGLE CLOUD PLATFORM  ##################
GCP_PROJECT_ID = os.environ['GCP_PROJECT']
GCP_SERVICE_ACCOUNT=os.environ['GCP_SERVICE_ACCOUNT']
GCP_SERVICE_ACCOUNT_KEY = os.environ['GCP_SERVICE_ACCOUNT_KEY']

##################  GCP BIG QUERY  ##################
BQ_REGION=os.environ['BQ_REGION']
BQ_SOURCE_DATASET = os.environ['BQ_SOURCE_DATASET']
BQ_CLEANED_DATASET = os.environ['BQ_CLEANED_DATASET']

##################  DATA SOURCES  ##################
DATA_SOURCES = ['national_polls', 'national_results', 'constituency_results',
                'national_google_trends', 'national_wikipedia', 'national_reddit']
DATA_RETRIEVAL = {
    'national_polls': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    'national_results': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    'constituency_results': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    'national_google_trends': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    'national_wikipedia': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    'national_reddit': {
        'dataset': '',
        'table': '',
        'query': f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    }
}
