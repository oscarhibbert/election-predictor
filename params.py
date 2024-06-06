import os

##################  GOOGLE CLOUD PLATFORM  ##################
GCP_PROJECT_ID = os.environ['GCP_PROJECT']
GCP_SERVICE_ACCOUNT=os.environ['GCP_SERVICE_ACCOUNT']
GCP_SERVICE_ACCOUNT_KEY = os.environ['GCP_SERVICE_ACCOUNT_KEY']
##################  GCP BIG QUERY  ##################
BQ_REGION=os.environ['BQ_REGION']
BQ_SOURCE_DATASET = os.environ['BQ_SOURCE_DATASET']
BQ_CLEANED_DATASET = os.environ['BQ_CLEANED_DATASET']
