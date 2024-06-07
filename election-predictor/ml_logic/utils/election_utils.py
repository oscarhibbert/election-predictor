import pandas as pd

def find_next_election_date(start_date, election_dates) -> pd.NaT:
    """
    Calculates time until next election.
    """
    for election_date in election_dates:
        if start_date < election_date:
            return election_date
    return pd.NaT
