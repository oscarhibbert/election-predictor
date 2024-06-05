import pandas as pd
# Imports polling data and filters for test years
polling_df = pd.read_csv('../processed_data/2004_to_2019_polling_cleaned.csv')
# Filters to 2019 election
election_2019_data = polling_df[polling_df['next_elec_date'] == '2019-12-12']
