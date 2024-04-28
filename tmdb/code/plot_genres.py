import pickle
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Load in the embeddings
with open('embeddings/movie_descriptions.pkl', 'rb') as f:
    movie_data = pickle.load(f)

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