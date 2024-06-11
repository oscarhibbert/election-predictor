import os

##################  GOOGLE CLOUD PLATFORM  ##################
GCP_PROJECT_ID = os.environ["GCP_PROJECT"]
GCP_SERVICE_ACCOUNT=os.environ["GCP_SERVICE_ACCOUNT"]
GCP_SERVICE_ACCOUNT_KEY = os.environ["GCP_SERVICE_ACCOUNT_KEY"]

##################  GCP BIG QUERY  ##################
BQ_REGION=os.environ["BQ_REGION"]
BQ_SOURCE_DATASET = os.environ["BQ_SOURCE_DATASET"]
BQ_CLEANED_DATASET = os.environ["BQ_CLEANED_DATASET"]

##################  DATA SOURCES  ##################
DATA_SOURCES_START_DATE = "2004-01-01"
DATA_SOURCES = ["national_polls", "national_results", "constituency_results",
                "national_google_trends", "national_wikipedia", "national_reddit",
                "ons_economic_data"]
DATA_RETRIEVAL = {
    "national_polls_results_combined": {
        "dataset": "master_sources",
        "table": "national_polls_results_combined",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    "national_polls": {
        "dataset": "master_sources",
        "table": "national_polls",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
        "feature_selection": ['samplesize', 'days_to_elec', 'poll_length',
                              'rating', 'BRX_FC', 'CON_FC', 'GRE_FC', 'LAB_FC',
                              'LIB_FC', 'NAT_FC', 'OTH_FC', 'PLC_FC', 'SNP_FC', 'UKI_FC']
    },
    "national_results": {
        "dataset": "master_sources",
        "table": "national_results",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    "constituency_results": {
        "dataset": "master_sources",
        "table": "constinuency_results",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table"
    },
    "constituency_bias": {
        "dataset": "master_sources",
        "table": "constituency_bias",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
    },
    "national_google_trends": {
        "dataset": "master_sources",
        "table": "national_google_trends",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
        "feature_selection": ["LAB_trends", "CON_trends", "LIB_trends",
                              "GRE_trends", "BRX_trends", "PLC_trends",
                              "SNP_trends", "UKI_trends", "NAT_trends"]
    },
    "national_wikipedia": {
        "dataset": "master_sources",
        "table": "national_wikipedia",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
        "feature_selection": ""
    },
    "national_reddit": {
        "dataset": "master_sources",
        "table": "national_reddit",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
        "feature_selection": ""
    },
    "ons_economic_data": {
        "dataset": "master_sources",
        "table": "ons_economic_data",
        "query": f"SELECT * FROM {GCP_PROJECT_ID}.dataset.table",
        "feature_selection": ["GDP", "Inflation", "Unemployment"]
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
    "Populus": "D+",
    "ICM": "D+",
    "IpsosMORI": "A-",
    "YouGov": "A-",
    "SavantaComRes": "B+",
    "BPIX": "F",
    "AngusReid": "F",
    "Harris": "C-",
    "TNSBMRB": "D",
    "Opinium": "A-",
    "Survation": "A-",
    "LordAshcroft": "D-",
    "Panelbase": "A-",
    "BMG": "B",
    "ORB": "D+",
    "Kantar": "B+",
    "Deltapoll": "D+",
    "NumberCruncherPolitics": "D",
    "Focaldata": "D+",
    "RedfieldWilton": "D",
    "JLPartners": "D",
    "FindOutNow": "D",
    "Omnisis": "D",
    "Techne": "D",
    "PeoplePolling": "D",
    "MoreinCommon": "F"
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
