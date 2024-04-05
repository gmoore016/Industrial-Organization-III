import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.manifold import TSNE
import numpy as np

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