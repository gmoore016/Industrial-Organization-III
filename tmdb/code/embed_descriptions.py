from openai import OpenAI, BadRequestError
import pandas as pd
import pickle
from tqdm import tqdm

tqdm.pandas()

# Input OpenAI API key
OPENAI_API_KEY = open('openai.secret', 'r').read()

def get_embedding(description):
    """
    Function taking in a description and returnning an embedding object
    """
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    if not description:
        print("No description!")
        return None

    try:
        response = client.embeddings.create(
            input=[description],
            model="text-embedding-3-small",
        )
    except:
        print("Could not process description!")
        print("Description:", description)
        raise

    return response

# Get descriptions
movie_data = pd.read_parquet('descriptions/movie_descriptions.parquet')

# Drop if descriptions are null
movie_data = movie_data.dropna(subset=['tmdb_description'])

# Embed descriptions
movie_data['embedding'] = movie_data['tmdb_description'].progress_apply(get_embedding)

with open('embeddings/movie_descriptions.pkl', 'wb') as f:
    pickle.dump(movie_data, f)