import tmdbsimple as tmdb
import pandas as pd
import csv
import pickle
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI


########## Pull data

# Read API key from file
with open('tmdb.secret', 'r') as f:
    api_key = f.read().strip()
    tmdb.API_KEY = api_key

# Get movie list from OpusData sample
opusdata = pd.read_csv('../OpusData/MovieData.csv')

print(opusdata.head())
# Get name and year of first movie
movie_name = opusdata['movie_name'][0]
movie_year = opusdata['production_year'][0]

search = tmdb.Search()
response = search.movie(query=movie_name, year=movie_year)

# Ensure only one result is found
assert search.total_results > 0, "No results found for movie"
assert search.total_results == 1, "Multiple results found for movie"

movie_info = search.results[0]
movie_id = movie_info['id']
print(movie_info)

TARANTINO = [466272, 24, 16869, 68718, 273248]
PIXAR = [862, 585, 12, 9806, 10681]
TWINS = [
    50544, # No Strings Attached
    41630, # Friends with Benefits
    20352, # Despicable Me
    38055, # Megamind
    374720, # Dunkirk
    399404, # Darkest Hour
]



# Get movie descriptions
def get_movie_descriptions(movie_ids):
    descriptions = []
    for movie_id in movie_ids:
        movie = tmdb.Movies(movie_id)
        response = movie.info()
        descriptions.append(response['overview'])
    return descriptions

# Get descriptions for Tarantino movies
tarantino_descriptions = get_movie_descriptions(TARANTINO)

# Write descriptions to file
with open('descriptions/tarantino.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    for i, description in enumerate(tarantino_descriptions):
        writer.writerow([TARANTINO[i], description, "tarantino"])

# Get descriptions for Pixar movies
pixar_descriptions = get_movie_descriptions(PIXAR)

# Write descriptions to file
with open('descriptions/pixar.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    for i, description in enumerate(pixar_descriptions):
        writer.writerow([PIXAR[i], description, "pixar"])

# Get descriptions for twin movies
twin_descriptions = get_movie_descriptions(TWINS)

# Write descriptions to file
with open('descriptions/twins.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    counter = 0
    for i, description in enumerate(twin_descriptions):
        writer.writerow([TWINS[i], description, "twin " + str(counter // 2)])
        counter += 1


######### Create embeddings


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



################ Plot t-SNE

# Load Tarantino embeddings
with open('embeddings/tarantino.pkl', 'rb') as f:
    tarantino = pickle.load(f)

# Load Pixar embeddings 
with open('embeddings/pixar.pkl', 'rb') as f:
    pixar = pickle.load(f)

# Load twin embeddings
with open('embeddings/twins.pkl', 'rb') as f:
    twins = pickle.load(f)

# Concatenate all embeddings
embeddings = pd.concat([tarantino, pixar, twins])

# Create numpy array from embeddings
X = embeddings['embedding'].apply(lambda x: x.data[0].embedding).to_list()
X = np.array(X)

# Create t-SNE model
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
reduced_descriptions = tsne.fit_transform(X)

# Add t-SNE results to dataframe
embeddings['x'] = reduced_descriptions[:, 0]
embeddings['y'] = reduced_descriptions[:, 1]

# Plot t-SNE results
plt.figure(figsize=(10, 10))
for group in embeddings['group'].unique():
    group_df = embeddings[embeddings['group'] == group]
    plt.scatter(group_df['x'], group_df['y'], label=group)
plt.legend()
plt.title('t-SNE of movie descriptions')

# Save plot to file
plt.savefig('plots/tsne.png')