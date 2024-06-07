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

##################  ELECTIONS  ##################
UK_ELECTIONS = {
    "2005": {
        "date": "2005-05-05",
    },
    "2010": {
        "date": "2010-05-06",
    },
    "2015": {
        "date": "2015-05-07",
    },
    "2017": {
        "date": "2017-06-08",
    },
    "2019": {
        "date": "2019-12-12",
    },
    "2024": {
        "date": "2024-07-04",
    }
}

UK_ELECTION_COUNTRY_CODES = {
    "scotland": "SCO",
    "great_britain": "GBR",
    "wales": "WAL",
    "northern_ireland": "NIR",
    "united_kingdom": "UKM",
    "england_wales": "EAW"
}

##################  POLLSTERS  ##################
POLLSTER_RATINGS = {
    'Populus': 'D+',
    'ICM': 'D+',
    'IpsosMORI': 'A-',
    'YouGov': 'A-',
    'SavantaComRes': 'B+',
    'BPIX': 'F',
    'AngusReid': 'F',
    'Harris': 'C-',
    'TNSBMRB': 'D',
    'Opinium': 'A-',
    'Survation': 'A-',
    'LordAshcroft': 'D-',
    'Panelbase': 'A-',
    'BMG': 'B',
    'ORB': 'D+',
    'Kantar': 'B+',
    'Deltapoll': 'D+',
    'NumberCruncherPolitics': 'D',
    'Focaldata': 'D+',
    'RedfieldWilton': 'D',
    'JLPartners': 'D',
    'FindOutNow': 'D',
    'Omnisis': 'D',
    'Techne': 'D',
    'PeoplePolling': 'D',
    'MoreinCommon': 'F'
}

##################  MODEL DEFAULT PARAMETERS  ##################
XGBOOST_PARAMS = {
    "learning_rate": 0.3,
    "n_estimators": 300,
    "max_depth": 3,
    "subsample": 0.7,
    "objective":"reg:squarederror",
    "nthread": -1,
    "enable_categorical": True
}
