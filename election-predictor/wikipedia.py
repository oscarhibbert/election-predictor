import requests
import pandas as pd
import datetime
import time

def get_wikipedia_pageviews(start_date, end_date, articles):
    """
    Fetches Wikipedia pageviews data for given articles between start_date and end_date.

    Parameters:
    - start_date (str): The start date in format YYYYMMDD.
    - end_date (str): The end date in format YYYYMMDD.
    - articles (list): List of article names to retrieve data for.

    Returns:
    - DataFrame: A DataFrame with date, article, and views columns.
    """
    base_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user'
    headers = {'User-Agent': 'YourAppName/1.0 (your.email@example.com)'}

    all_data = []

    for article in articles:
        url = f"{base_url}/{article}/monthly/{start_date}/{end_date}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json().get('items', [])
            for item in data:
                all_data.append({
                    'article': article,
                    'date': item['timestamp'][:8],  # Extract date from timestamp
                    'views': item['views']
                })
        else:
            print(f"Failed to retrieve data for {article}: {response.status_code} - {response.text}")

        # Sleep to avoid hitting the API rate limit
        time.sleep(1)

    if all_data:
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        return df
    else:
        print("No data was retrieved.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Define the date range
    start_date = '20150701'  # Start date in format YYYYMMDD
    end_date = datetime.datetime.now().strftime('%Y%m%d')  # Today's date

    # List of articles to retrieve data for
    articles = ['Labour Party (UK)', 'Conservative Party (UK)']

    # Fetch the data
    df_pageviews = get_wikipedia_pageviews(start_date, end_date, articles)

    # Display the data
    print(df_pageviews.head())
