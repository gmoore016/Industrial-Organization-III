from openai import OpenAI
import pandas as pd
import pickle

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

    response = client.embeddings.create(
        input=[description],
        model="text-embedding-3-small"
    )

    return response

# Get descriptions
movie_data = pd.read_csv('descriptions/movie_descriptions.csv', encoding="ISO-8859-1")

movie_data['embedding'] = movie_data['tmdb_description'].apply(get_embedding)

with open('embeddings/movie_descriptions.pkl', 'wb') as f:
    pickle.dump(movie_data, f)