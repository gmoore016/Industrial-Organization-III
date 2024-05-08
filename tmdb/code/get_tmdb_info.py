import tmdbsimple as tmdb
from unidecode import unidecode
import pandas as pd
import string
import csv
from tqdm import tqdm
import re

PAREN_REGEX = re.compile(r'\([^)]*\)')


def get_movie_info(row):
    """
    Function to apply to each row in the movie data
    Takes info from that row and returns the movie info from TMDB
    """
    cleaned_name = row['Title']
    year = row['year']

    # Flag rereleases to change search year constraints
    rerelease = False
    if "RE" in cleaned_name or "rerelease" in cleaned_name.lower() or "reissue" in cleaned_name.lower() or "re-release" in cleaned_name.lower():
        cleaned_name = cleaned_name.replace("RE", "")
        cleaned_name = cleaned_name.replace("rerelease", "")
        cleaned_name = cleaned_name.replace("reissue", "")
        cleaned_name = cleaned_name.replace("re-release", "")
        rerelease = True

    # Remove anything in parentheses
    cleaned_name = re.sub(PAREN_REGEX, '', cleaned_name)

    # Remove anything after colon
    cleaned_name = cleaned_name.split(":")[0]

    # Remove accents
    cleaned_name = unidecode(cleaned_name)

    # Remove punctuation
    cleaned_name = cleaned_name.translate(str.maketrans('', '', string.punctuation))

    # Need to wait until after above to successfully flag RE
    cleaned_name = cleaned_name.lower()
    if "imax" in cleaned_name:
        cleaned_name = cleaned_name.replace(" imax", " ")
    cleaned_name = cleaned_name.replace(" pt", " Part")
                                        
    
    
    

    # Search for movie given name and year
    search = tmdb.Search()
    if rerelease:
        search.movie(query=cleaned_name)
    else:
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
tqdm.pandas()
query_results = movies.progress_apply(get_movie_info, axis=1)

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