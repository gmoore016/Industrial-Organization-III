import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import re 

GENRE_REGEX = re.compile(r"\[(\d+)")

# Load in the embeddings
with open('embeddings/movie_descriptions.pkl', 'rb') as f:
    movie_data = pickle.load(f)

movie_data['genre'] = movie_data['tmdb_genres'].apply(lambda x: re.search(GENRE_REGEX, x).group(1) if re.search(GENRE_REGEX, x) else None)
movie_data['genre'] = movie_data['genre'].astype(pd.Int64Dtype())

# Label genre integers
genre_labels = {
    28: 'Action',
    12: 'Adventure',
    16: 'Animation',
    35: 'Comedy',
    80: 'Crime',
    99: 'Documentary',
    18: 'Drama',
    10751: 'Family',
    14: 'Fantasy',
    36: 'History',
    27: 'Horror',
    10402: 'Music',
    9648: 'Mystery',
    10749: 'Romance',
    878: 'Science Fiction',
    10770: 'TV Movie',
    53: 'Thriller',
    10752: 'War',
    37: 'Western'
}

movie_data['genre'] = movie_data['genre'].map(genre_labels)

X = movie_data['embedding'].apply(lambda x: x.data[0].embedding).to_list()
X = np.array(X)

# Create t-SNE model
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
reduced_descriptions = tsne.fit_transform(X)

# Add t-SNE results to dataframe
movie_data['x'] = reduced_descriptions[:, 0]
movie_data['y'] = reduced_descriptions[:, 1]

# Plot t-SNE results
plt.figure(figsize=(10, 10))

# Plot each genre in a different color
for genre in movie_data['genre'].unique():
    genre_data = movie_data[movie_data['genre'] == genre]
    plt.scatter(genre_data['x'], genre_data['y'], label=genre)
plt.legend()

# Save the plot
plt.savefig('plots/tsne_genres.png')