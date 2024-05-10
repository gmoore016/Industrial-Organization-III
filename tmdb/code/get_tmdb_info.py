import tmdbsimple as tmdb
from unidecode import unidecode
import pandas as pd
import string
from tqdm import tqdm
import re

PAREN_REGEX = re.compile(r'\([^)]*\)')


def get_movie_info(row):
    """
    Function to apply to each row in the movie data
    Takes info from that row and returns the movie info from TMDB
    """
    cleaned_name = row["Clean Title"]
    year = row["year"]
    rerelease = row["rerelease"]
    
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

movies = pd.read_parquet('../guru/data/movies.parquet')
movies = movies.reset_index(drop=False)

# Get the first date of each movie
movies['year'] = movies['Wide Release Date'].year


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
movies.to_parquet('descriptions/movie_descriptions.parquet', index=False)