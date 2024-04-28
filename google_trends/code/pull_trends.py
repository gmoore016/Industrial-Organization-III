from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt

# Create pytrends object
pytrends = TrendReq(hl='en-US', tz=360)

# Load data from OpusData
opusdata = pd.read_csv('../OpusData/MovieData.csv')

# Get list of movie names
movie_names = opusdata['movie_name'].to_list()
movie_years = opusdata['production_year'].to_list()
movie_ids = opusdata['movie_odid'].to_list()

# Create dataframe to store movie interests
# Has three columns: movie id, date, and interest
movie_interests_long = pd.DataFrame(columns=['movie_id', 'date', 'interest'])

for i in range(len(movie_names)):
    # Unpack movie data
    movie_name = movie_names[i]
    movie_year = movie_years[i]
    movie_id = movie_ids[i]

    # Get Google Trends data
    pytrends.build_payload(
        kw_list=[movie_name + " movie"],
        timeframe=f'{movie_year - 1}-01-01 {movie_year + 1}-12-31',
    )

    # Get interest over time
    interest_over_time = pytrends.interest_over_time()

    # Drop isPartial column
    try:
        interest_over_time = interest_over_time.drop(columns='isPartial')
    except KeyError:
        print(f"Error with {movie_name} {movie_year}")
        print(interest_over_time)

    # Drop if value is zero
    interest_over_time = interest_over_time[interest_over_time[movie_name + ' movie'] != 0]

    # Add movie id to column
    interest_over_time['movie_id'] = movie_id

    # Reset index
    interest_over_time = interest_over_time.reset_index()

    # Rename columns
    interest_over_time.columns = ['date', 'interest', 'movie_id']

    # Add results to end of long dataframe
    movie_interests_long = pd.concat([movie_interests_long, interest_over_time])

# Save movie_interests_long
movie_interests_long.to_csv('output/movie_interests_long.csv', index=False)

