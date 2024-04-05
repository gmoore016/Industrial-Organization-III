import tmdbsimple as tmdb
from unidecode import unidecode
import pandas as pd
import string
import csv

def clean_name(name):
    """
    Function to clean movie name
    """
    if not name:
        return None 
    
    # Remove accents and asian characters
    clean_name = unidecode(clean_name)
    
    # Remove punctuation
    clean_name = name.translate(str.maketrans('', '', string.punctuation))

    # Lowercase
    clean_name = clean_name.lower()

    # Handcoding some exceptions
    if clean_name == 'chang jiang qi hao  cj7':
        clean_name = 'cj7'
    if clean_name == 'stan helsing a parody':
        clean_name = 'stan helsing'
    elif clean_name == 'san suk si gin':
        clean_name = 'shinjuku incident'
    elif clean_name == 'shi yue wei cheng':
        clean_name = 'bodyguards and assassins'
    elif clean_name == 'les intouchables':
        clean_name = 'the intouchables'
    elif clean_name == 'lee daniels the butler':
        clean_name = 'the butler'
    elif clean_name == 'mr popperss penguins':
        clean_name = 'mr poppers penguins'
    elif clean_name == 'doctor seuss the lorax':
        clean_name = 'the lorax'
    elif clean_name == 'jin ling shi san chai':
        clean_name = 'the flowers of war'
    elif clean_name == 'san cheng ji':
        clean_name = 'a tale of three cities'
    elif clean_name == 'savva serdtse voyna':
        clean_name = 'savva heart of the warrior'
    elif clean_name == 'star wars ep vii the force awakens':
        clean_name = 'star wars the force awakens'
    elif clean_name == 'chai dan zhuanjia':
        clean_name = 'shock wave'
    elif clean_name == 'star wars ep viii the last jedi':
        clean_name = 'star wars the last jedi'
    elif clean_name == 'spiderman into the spiderverse 3d':
        clean_name = 'spiderman into the spiderverse'
    elif clean_name == 'walle':
        clean_name = 'WALL·E'
    elif clean_name == 'halloween 2':
        clean_name = 'halloween ii'
    elif clean_name == 'michael jacksons this is it':
        clean_name = 'this is it'
    elif clean_name == 'disneys a christmas carol':
        clean_name = 'a christmas carol'

    return clean_name

def get_movie_info(row):
    """
    Function to apply to each row in the movie data
    Takes info from that row and returns the movie info from TMDB
    """

    # First we'll clean the movie name
    # Remove punctuation
    cleaned_name = clean_name(row['movie_name'])

    year = row['production_year']

    if cleaned_name == 'romance and cigarettes':
        year = 2005
    if cleaned_name == 'slow burn':
        year = 2005
    if cleaned_name == 'red tails':
        year = 2012
    elif cleaned_name == 'august osage county':
        year = 2013
    elif cleaned_name == 'sin city a dame to kill for':
        year = 2014
    elif cleaned_name == 'edge of tomorrow':
        year = 2014
    elif cleaned_name == 'the last song':
        year = 2010
    elif cleaned_name == 'faster':
        year = 2010
    elif cleaned_name == "valentines day":
        year = 2010
    elif cleaned_name == 'killers':
        year = 2010
    elif cleaned_name == 'the roommate':
        year = 2011
    elif cleaned_name == 'the beaver':
        year = 2011
    elif cleaned_name == 'the help':
        year = 2011
    elif cleaned_name == 'the eagle':
        year = 2011
    elif cleaned_name == 'no strings attached':
        year = 2011
    elif cleaned_name == 'hanna':
        year = 2011
    elif cleaned_name == 'winnie the pooh':
        year = 2011
    elif cleaned_name == 'sanctum':
        year = 2011
    elif cleaned_name == 'the tree of life':
        year = 2011
    elif cleaned_name == 'abduction':
        year = 2011

    # Search for movie given name and year
    search = tmdb.Search()
    search.movie(query=cleaned_name, primary_release_year=year)

    if not search.total_results:
        #print(f"Movie {row['movie_name']} not found in year {row['production_year']}, trying previous year")
        search.movie(query=cleaned_name, primary_release_year= year - 1)

    if not search.total_results:
        #print(f"Movie {row['movie_name']} not found in year {row['production_year']}, trying next year")
        search.movie(query=cleaned_name, primary_release_year = year + 1)

    if not search.total_results:
        #print(f"Movie {row['movie_name']} not found in year {row['production_year']}, trying two years out")
        search.movie(query=cleaned_name, primary_release_year = year + 2)

    # If no results
    if not search.total_results:
        print(f"No results found for {cleaned_name} around year {year}")
        return None

    return search.results[0]


# Read API key from file
with open('tmdb.secret', 'r') as f:
    api_key = f.read().strip()
    tmdb.API_KEY = api_key

# Get movie list from OpusData sample
opusdata = pd.read_csv('../OpusData/MovieData.csv')

# Get movie info for each movie
query_results = opusdata.apply(get_movie_info, axis=1)

opusdata['tmdb_id'] = [query_result['id'] if query_result else None for query_result in query_results]
opusdata['tmdb_name'] = [query_result['title'] if query_result else None for query_result in query_results]
opusdata['tmdb_description'] = [query_result['overview'] if query_result else None for query_result in query_results]
opusdata['tmdb_release_date'] = [query_result['release_date'] if query_result else None for query_result in query_results]
opusdata['tmdb_genres'] = [query_result['genre_ids'] if query_result else None for query_result in query_results]
opusdata['tmdb_language'] = [query_result['original_language'] if query_result else None for query_result in query_results]

# Flag those where clean names do not match
opusdata['name_mismatch'] = opusdata['movie_name'].apply(clean_name) != opusdata['tmdb_name'].apply(clean_name)

# Flag those where no results found
opusdata['no_results'] = opusdata['tmdb_id'].isnull()

# Write descriptions to file
opusdata.to_csv('descriptions/movie_descriptions.csv', index=False)