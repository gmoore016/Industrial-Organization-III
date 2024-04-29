from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from requests.exceptions import RetryError

# Create pytrends object
pytrends = TrendReq(
    hl='en-US', 
    tz=360,
    retries=3,
)

# Load data from OpusData
opusdata = pd.read_csv('../OpusData/output/opus_cleaned.csv')

# Get list of movie names
movie_names = opusdata['cleaned_name'].to_list()
movie_years = opusdata['production_year'].to_list()
movie_ids = opusdata['movie_odid'].to_list()

# Get list of movie ids already scraped
existing_output = os.listdir('raw')

for i in range(len(movie_names)):
    # Unpack movie data
    movie_name = movie_names[i]
    movie_year = movie_years[i]
    movie_id = movie_ids[i]

    # Skip if already scraped
    filename = f'{movie_id}.csv'
    if filename in existing_output:
        continue

    # Get Google Trends data
    try:
        pytrends.build_payload(
            kw_list=[movie_name + " movie"],
            timeframe=f'{movie_year - 1}-01-01 {movie_year + 1}-12-31',
        )

        # Get interest over time
        interest_over_time = pytrends.interest_over_time()
    except RetryError:
        # Too many requests; wait a while and try again
        print('Too many requests; waiting 60 seconds')
        time.sleep(60)
        pytrends.build_payload(
            kw_list=[movie_name + " movie"],
            timeframe=f'{movie_year - 1}-01-01 {movie_year + 1}-12-31',
        )

        # Get interest over time
        interest_over_time = pytrends.interest_over_time()
    

    # If no results, flag error and continue
    if interest_over_time.empty:
        print(f'No results found for {movie_name}')
        interest_over_time = pd.DataFrame(columns=['date', 'interest', 'movie_id'])
        interest_over_time['movie_id'] = movie_id
        interest_over_time.to_csv(f'raw/{filename}', index=False)
        continue

    # Drop isPartial column
    interest_over_time = interest_over_time.drop(columns='isPartial')

    # Drop if value is zero
    interest_over_time = interest_over_time[interest_over_time[movie_name + ' movie'] != 0]

    # Add movie id to column
    interest_over_time['movie_id'] = movie_id

    # Reset index
    interest_over_time = interest_over_time.reset_index()

    # Rename columns
    interest_over_time.columns = ['date', 'interest', 'movie_id']

    # Write results to file
    interest_over_time.to_csv(f'raw/{filename}', index=False)


