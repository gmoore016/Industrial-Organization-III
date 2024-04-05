from openai import OpenAI
import csv
import pandas as pd
import pickle

# Fetch API key from secret file
OPENAI_API_KEY = open('openai.secret', 'r').read()

def get_embedding(description):

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    response = client.embeddings.create(
        input=[description],
        model="text-embedding-ada-002"
    )

    return response

# Read Tarantino movie descriptions into dataframe
tarantino = pd.read_csv('descriptions/tarantino.csv', encoding="ISO-8859-1")
tarantino['embedding'] = tarantino['description'].apply(get_embedding)

# Save to disk to avoid re-embedding
with open('embeddings/tarantino.pkl', 'wb') as f:
    pickle.dump(tarantino, f)

# Read Pixar movie descriptions into dataframe
pixar = pd.read_csv('descriptions/pixar.csv', encoding="ISO-8859-1")
pixar['embedding'] = pixar['description'].apply(get_embedding)

# Save to disk to avoid re-embedding
with open('embeddings/pixar.pkl', 'wb') as f:
    pickle.dump(pixar, f)

# Read twin movie descriptions into dataframe
twins = pd.read_csv('descriptions/twins.csv', encoding="ISO-8859-1")
twins['embedding'] = twins['description'].apply(get_embedding)

# Save to disk to avoid re-embedding
with open('embeddings/twins.pkl', 'wb') as f:
    pickle.dump(twins, f)