import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import minimize
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
import linearmodels as lm
import time
import statsmodels.formula.api as smf

N_DIMS = 6

# Load the embeddings from parquet
tmdb = pd.read_parquet('../tmdb/embeddings/movie_descriptions.parquet')

# Drop if embeddings are null
tmdb = tmdb.dropna(subset=['embedding'])
tmdb = tmdb.reset_index(drop=True)

# Get array of movie ids
movie_ids = tmdb['movie_id'].copy()

raw_embeddings = np.array(tmdb['embedding'].values)
raw_embeddings = np.array([np.array(x) for x in raw_embeddings])

# Create distance matrix by taking dot product of embeddings
distances = 1 - raw_embeddings @ raw_embeddings.T

# Normalize distances
distances = distances / np.max(distances)

#reduced_embeddings = PCA(n_components=N_DIMS).fit_transform(raw_embeddings)

#embedding_frame = pd.DataFrame(reduced_embeddings, columns=[f'pca_{i}' for i in range(N_DIMS)])

#for i in range(N_DIMS):
#    embeddings[f'pca_{i}'] = reduced_embeddings[:, i]

# Load box office info
guru = pd.read_parquet('../guru/data/weekly_sales.parquet')

guru = guru.reset_index(drop=False)

print(guru.dtypes)

# Compute log interest
guru['ln_earnings'] = np.log(guru['Week Sales'])

# Drop if missing weeks
guru = guru.dropna(subset=['Weeks'])

# Histogram of weeks since release
plt.hist(guru['Weeks'], bins=20)
plt.xlabel('Weeks Since Wide Release')
plt.ylabel('Frequency')
plt.title('Histogram of Weeks Since Wide Release')
plt.savefig('output/weeks_histogram.png')

# Limit sample to movies with trends and embeddings
movies_with_trends = guru['movie_id'].unique()
movies_with_embeddings = tmdb['movie_id'].unique()
movies = np.intersect1d(movies_with_trends, movies_with_embeddings)

assert len(movies) == distances.shape[0]

print(f'Number of movies satisfying all criteria: {len(movies)}')

guru = guru[guru['movie_id'].isin(movies)]
embeddings = tmdb[tmdb['movie_id'].isin(movies)]

# Get the index of distances at bottom and top ends
#empirical_distances = np.array(list(movie_distances.values()))
bottom_end = np.percentile(distances, 10)
top_end = np.percentile(distances, 90)

# Find the closest movies for a subsample of the data
examples = [
    "e487ee1bc1477d6c2828d91e94c01c6e_0_1_2_3_4_5_6", # Tangled
    "72858d1af3c55029d5dc0bf1a77b6d9e_0_1_2_3_4_5_6", # Frozen
    "cabdb1014252d39ac018f447e7d5fbc2_0_1_2_3_4_5_6", # Dune: Part 1
    "31a3df9bd28be76c134d369e990d5094_0_1_2_3_4_5_6", # The Notebook
    "90eb7723a657b6597100aafef171d9f2_2_3_4_5_6", # Avengers: Endgame
    "635d8ed0689c0a83f596cf04ebe45b97_0_1_2_3_4_5_6", # A Bug's Life
]

# For each example, find the two closest movies
for example in examples:
    # Get index of the example
    example_index = np.where(movie_ids == example)[0][0]

    example_name = tmdb[tmdb['movie_id'] == example]['Clean Title'].values[0]

    distances_to_example = distances[example_index, :].copy()

    print("\n")

    print(f'Closest movies to {example_name}:')
    print("-------------------------")
    for i in range(10):
        closest_index = np.argmin(distances_to_example)
        closest_movie = movie_ids[closest_index]
        closest_distance = distances_to_example[closest_index]

        closest_movie_name = tmdb['Clean Title'][closest_index]

        print(f'{closest_movie_name}: {closest_distance}')
        distances_to_example[closest_index] = np.inf

    print("\n")

    distances_to_example = distances[example_index].copy()
    print("Movies which are close:")
    distances_from_bottom_decile = np.abs(distances_to_example - bottom_end)
    for i in range(10):
        closest_index = np.argmin(distances_from_bottom_decile)
        closest_movie = movie_ids[closest_index]
        closest_distance = distances_to_example[closest_index]

        closest_movie_name = tmdb[tmdb['movie_id'] == closest_movie]['Clean Title'].values[0]

        print(f'{closest_movie_name}: {closest_distance}')
        distances_from_bottom_decile[closest_index] = np.inf

    print("\n")

    distances_to_example = distances[example_index].copy()
    print("Movies which are far:")
    distances_from_top_decile = np.abs(distances_to_example - top_end)
    for i in range(10):
        closest_index = np.argmin(distances_from_top_decile)
        closest_movie = movie_ids[closest_index]
        closest_distance = distances_to_example[closest_index]

        closest_movie_name = tmdb[tmdb['movie_id'] == closest_movie]['Clean Title'].values[0]

        print(f'{closest_movie_name}: {closest_distance}')
        distances_from_top_decile[closest_index] = np.inf

# For each movie-date, compute the leave-one-out set of distances within the date
gamma0 = []
gamma1 = []
gamma2 = []
gamma3 = []

for row in guru.itertuples():
    movie_id = row.movie_id
    date = row.Date

    # Get index of movie_id in movie_ids
    movie_index = np.where(movie_ids == movie_id)[0][0]

    # Get all movies on the date
    date_movies = guru[guru['Date'] == date]
    
    competitor_ids = date_movies['movie_id'].unique()

    # Compute the distances
    competitor_distances = []
    for comparison_movie in competitor_ids:
        if movie_id == comparison_movie:
            continue

        comparison_index = np.where(movie_ids == comparison_movie)[0][0]

        if np.isinf(distances[movie_index, comparison_index]):
            print(f'Infinite distance between {movie_id} and {comparison_movie}')

        competitor_distances.append(distances[movie_index, comparison_index])

    competitor_distances = np.array(competitor_distances)

    # No need to weight by log price since log price is constant
    gamma0.append(len(competitor_distances))
    gamma1.append(np.sum(competitor_distances))
    gamma2.append(np.sum(competitor_distances ** 2))
    gamma3.append(np.sum(competitor_distances ** 3))

# Add to the guru data
guru['gamma0'] = gamma0
guru['gamma1'] = gamma1
guru['gamma2'] = gamma2
guru['gamma3'] = gamma3

# Convert movie_id to categorical
guru['movie_id'] = guru['movie_id'].astype('category')

guru.set_index(['movie_id', 'Date'], inplace=True)

print(guru.head())

# Print rows containing infinite values
print(guru[guru.isin([np.nan, np.inf, -np.inf]).any(axis=1)])

# Run regression of log earnings on distances and age
X = guru[['ln_earnings', 'gamma0', 'gamma1', 'gamma2', 'gamma3', 'Weeks']]
X['Weeks'] = X['Weeks'].astype('category')
X = sm.add_constant(X)
formula = 'ln_earnings ~ gamma0 + gamma1 + gamma2 + gamma3 + C(Weeks) + EntityEffects + TimeEffects'
reg = lm.PanelOLS.from_formula(formula=formula, data=X, drop_absorbed=True)

results = reg.fit(low_memory=False)
print(results)

# Get coefficients on gamma1, gamma2, and gamma3
gamma0_coefficient = results.params['gamma0']
gamma1_coefficient = results.params['gamma1']
gamma2_coefficient = results.params['gamma2']
gamma3_coefficient = results.params['gamma3']

# Plot cross-elasticities over distance
distance_grid = np.linspace(0, 1, 100)
cross_elasticity = gamma0_coefficient + gamma1_coefficient * distance_grid + gamma2_coefficient * distance_grid ** 2 + gamma3_coefficient * distance_grid ** 3

bottom_index = np.argmin(np.abs(distance_grid - bottom_end))
top_index = np.argmin(np.abs(distance_grid - top_end))


print(f'Cross-elasticity at bottom end of distance: {cross_elasticity[bottom_index]}')
print(f'Cross-elasticity at top end of distance: {cross_elasticity[top_index]}')

# Get the vector of distances
#empirical_distances = np.array(list(movie_distances.values()))

# Flatten the distances
distances = distances.flatten()

# Plot data density and cross-elasticity on same plot
fig, ax1 = plt.subplots()

# Add title
ax1.set_title('Cross-Elasticity of Distance')
ax2 = ax1.twinx()

# Plot the empirical distances
ax2.hist(distances, bins=20, alpha=0.5, density=True)
ax2.set_xlabel('Distance')
ax2.set_ylabel('Density of Empirical Distance')

ax1.plot(distance_grid, cross_elasticity, color='orange')
ax1.set_ylabel('Cross-Elasticity of Distance')

ax2.axvline(x=top_end, color='red', linestyle='--')
ax2.axvline(x=bottom_end, color='red', linestyle='--')

fig.tight_layout()
plt.savefig('output/cross_elasticity.png')

# Zoom in on the cross-elasticity in the support of the data

fig, ax1 = plt.subplots()

# Limit only to bottom 10% to top 10%
ax1.set_xlim(bottom_end - .2, top_end + .2)

# Bound the y axis based on range between bottom and top end
ymin = np.min(cross_elasticity[bottom_index:top_index])
ymax = np.max(cross_elasticity[bottom_index:top_index])
margin = (ymax - ymin)
ax1.set_ylim(ymin - margin, ymax + margin)

# Add title
ax1.set_title('Cross-Elasticity of Distance')

ax1.plot(distance_grid, cross_elasticity, color='orange')
ax1.set_ylabel('Cross-Elasticity of Distance')

ax2 = ax1.twinx()

# Plot the empirical distances
ax2.hist(distances, bins=20, alpha=0.5, density=True)
ax2.set_xlabel('Distance')
ax2.set_ylabel('Density of Empirical Distance')

# Add vertical lines at the top and bottom ends
ax2.axvline(x=top_end, color='red', linestyle='--')
ax2.axvline(x=bottom_end, color='red', linestyle='--')

fig.tight_layout()
plt.savefig('output/cross_elasticity_zoom.png')
