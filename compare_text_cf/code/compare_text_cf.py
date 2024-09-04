import pandas as pd
import matplotlib.pyplot as plt

# Load the text similarities
text_similarities = pd.read_parquet('../movie_text_distances/output/movie_text_distances.parquet')
text_similarities = text_similarities.rename(columns={'distance': 'text_distance'})
text_similarities = text_similarities.sort_values(['tmdb_id_1', 'text_distance'], ascending=[True, True])
print(text_similarities.head())

collaborative_similarities = pd.read_csv('../movielens/output/UV_decomp_similarity.csv')
collaborative_similarities['collaborative_distance'] = 1 - collaborative_similarities['similarity']
collaborative_similarities = collaborative_similarities.rename(columns={'tmdbId_x': 'tmdb_id_1', 'tmdbId_y': 'tmdb_id_2'})
collaborative_similarities = collaborative_similarities[['tmdb_id_1', 'tmdb_id_2', 'collaborative_distance']]
collaborative_similarities = collaborative_similarities.sort_values(['tmdb_id_1', 'collaborative_distance'], ascending=[True, False])
print(collaborative_similarities.head())

# Merge the two similarity matrices
similarity = text_similarities.merge(collaborative_similarities, on=['tmdb_id_1', 'tmdb_id_2'], how='inner')

corr = similarity['text_distance'].corr(similarity['collaborative_distance'])

# Plot the two distances against each other
plt.scatter(similarity['text_distance'], similarity['collaborative_distance'], alpha=0.1)
plt.xlabel('Text Distance')
plt.ylabel('Collaborative Distance')
plt.title(f'Collaborative vs Text Distance (Correlation: {corr:.2f})')
plt.show()



