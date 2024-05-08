import tmdbsimple as tmdb
from unidecode import unidecode
import pandas as pd
import string
import csv


def get_movie_info(row):
    """
    Function to apply to each row in the movie data
    Takes info from that row and returns the movie info from TMDB
    """
    cleaned_name = row['Title']
    year = row['year']

    # Search for movie given name and year
    search = tmdb.Search()
    search.movie(query=cleaned_name, year=year)

    if not search.total_results:
        search.movie(query=cleaned_name, year= year - 1)

    if not search.total_results:
        search.movie(query=cleaned_name, year = year + 1)

    if not search.total_results:
        search.movie(query=cleaned_name, year = year + 2)

    # If no results
    if not search.total_results:
        print(f"No results found for {cleaned_name} around year {year}")
        return None

    return search.results[0]

# Read API key from file
with open('tmdb.secret', 'r') as f:
    api_key = f.read().strip()
    tmdb.API_KEY = api_key

gurudata = pd.read_csv('../guru/data/weekly_sales.csv')

# Convert Date to datetime
gurudata['Date'] = pd.to_datetime(gurudata['Date'])

# Get the first date of each movie
movies = gurudata.groupby('Title')['Date'].min().reset_index()
movies['year'] = movies['Date'].dt.year

# Merge premier onto movie data

# Get movie info for each movie
query_results = movies.apply(get_movie_info, axis=1)

# Parse TMDB info into dataframe
movies = movies.copy()
movies['tmdb_id'] = [query_result['id'] if query_result else None for query_result in query_results]
movies['tmdb_name'] = [query_result['title'] if query_result else None for query_result in query_results]
movies['tmdb_description'] = [query_result['overview'] if query_result else None for query_result in query_results]
movies['tmdb_release_date'] = [query_result['release_date'] if query_result else None for query_result in query_results]
movies['tmdb_genres'] = [query_result['genre_ids'] if query_result else None for query_result in query_results]
movies['tmdb_language'] = [query_result['original_language'] if query_result else None for query_result in query_results]

# Flag those where no results found
movies['no_results'] = movies['tmdb_id'].isnull()

# Write descriptions to file
movies.to_csv('descriptions/movie_descriptions.csv', index=False)