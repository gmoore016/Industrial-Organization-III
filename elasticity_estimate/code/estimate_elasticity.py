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
embeddings = pd.read_parquet('../tmdb/embeddings/movie_descriptions.parquet')

# Drop if embeddings are null
embeddings = embeddings.dropna(subset=['embedding'])

raw_embeddings = np.array(embeddings['embedding'].values)
raw_embeddings = np.array([np.array(x) for x in raw_embeddings])

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
movies_with_embeddings = embeddings['movie_id'].unique()
movies = np.intersect1d(movies_with_trends, movies_with_embeddings)

print(f'Number of movies satisfying all criteria: {len(movies)}')

guru = guru[guru['movie_id'].isin(movies)]
embeddings = embeddings[embeddings['movie_id'].isin(movies)]

# Create distances between relevant movies
movie_distances = {}

max_distance = 0

# Get the distribution of distances between movies
empirical_distances = []
for row1 in embeddings.itertuples():
    for row2 in embeddings.itertuples():
        if row1.movie_id == row2.movie_id:
            continue

        movie1 = row1.embedding
        movie2 = row2.embedding

        distance = cosine(movie1, movie2)

        empirical_distances.append(distance)

# Clear canvas
plt.clf()

# Plot histogram of distances
plt.hist(empirical_distances, bins=50)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Distances Between Movies')
plt.savefig('output/distance_histogram_unnormalized.png')

# For each date in the guru data, get the set of corresponding movies
for date in guru['Date'].unique():
    movies = guru[guru['Date'] == date]['movie_id'].unique()

    # For each pair of movies, compute the distance
    movie_pairs = [(movie_id1, movie_id2) for movie_id1 in movies for movie_id2 in movies if movie_id1 != movie_id2]

    for movie_id1, movie_id2 in movie_pairs:
        if (movie_id1, movie_id2) in movie_distances:
            continue
        else:
            movie1 = embeddings[embeddings['movie_id'] == movie_id1]
            movie2 = embeddings[embeddings['movie_id'] == movie_id2]

            # Filter only to pca columns
            movie1 = movie1['embedding'].values[0]
            movie2 = movie2['embedding'].values[0]

            distance = cosine(movie1, movie2)
            if distance > max_distance:
                max_distance = distance
            movie_distances[(movie_id1, movie_id2)] = distance
            movie_distances[(movie_id2, movie_id1)] = distance

# Normalize distances
for key, value in movie_distances.items():
    movie_distances[key] = value / max_distance

# Get the index of distances at bottom and top ends
#empirical_distances = np.array(list(movie_distances.values()))
empirical_distances = np.array(empirical_distances)
bottom_end = np.percentile(empirical_distances, 10)
top_end = np.percentile(empirical_distances, 90)

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
    example_embedding = embeddings[embeddings['movie_id'] == example]
    example_embedding = example_embedding['embedding'].values[0]

    distances = []
    for row in embeddings.itertuples():
        if row.movie_id == example:
            continue

        #movie_embedding = np.array([row.pca_0, row.pca_1, row.pca_2, row.pca_3, row.pca_4, row.pca_5])
        movie_embedding = row.embedding
        distance = cosine(example_embedding, movie_embedding)
        norm_distance = distance / max_distance
        distances.append((row.movie_id, row.tmdb_name, norm_distance))

    distances = sorted(distances, key=lambda x: x[2])
    print(f'Closest movies to {example}:')
    print("-------------------------")
    for i in range(10):
        print(f'{distances[i][1]}: {distances[i][2]} (movie_id: {distances[i][0]})')
    print("Movies which are close:")
    # Get a movie that is one end away
    sim_movies = [x for x in distances if x[2] < bottom_end][-5:]
    for sim_movie in sim_movies:
        print(f'{sim_movie[1]}: {sim_movie[2]} (movie_id: {sim_movie[0]})')
    print("Movies which are far:")
    # Get a movie that is nine ends away
    sim_movies = [x for x in distances if x[2] < top_end][-5:]
    for sim_movie in sim_movies:
        print(f'{sim_movie[1]}: {sim_movie[2]} (movie_id: {sim_movie[0]})')
    print("\n")

# For each movie-date, compute the leave-one-out set of distances within the date
gamma0 = []
gamma1 = []
gamma2 = []
gamma3 = []

for row in guru.itertuples():
    movie_id = row.movie_id
    date = row.Date

    # Get all movies on the date
    date_movies = guru[guru['Date'] == date]
    
    competitor_ids = date_movies['movie_id'].unique()

    # Compute the distances
    distances = []
    for comparison_movie in competitor_ids:
        if movie_id == comparison_movie:
            continue
        distances.append(movie_distances[(movie_id, comparison_movie)])

    distances = np.array(distances)

    # No need to weight by log price since log price is constant
    gamma0.append(len(distances))
    gamma1.append(np.sum(distances))
    gamma2.append(np.sum(distances ** 2))
    gamma3.append(np.sum(distances ** 3))

# Add to the guru data
guru['gamma0'] = gamma0
guru['gamma1'] = gamma1
guru['gamma2'] = gamma2
guru['gamma3'] = gamma3

# Convert movie_id to categorical
guru['movie_id'] = guru['movie_id'].astype('category')

guru.set_index(['movie_id', 'Date'], inplace=True)

print(guru.head())

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
distances = np.linspace(0, 1, 100)
cross_elasticity = gamma0_coefficient + gamma1_coefficient * distances + gamma2_coefficient * distances ** 2 + gamma3_coefficient * distances ** 3

bottom_index = np.argmin(np.abs(distances - bottom_end))
top_index = np.argmin(np.abs(distances - top_end))


print(f'Cross-elasticity at bottom end of distance: {cross_elasticity[bottom_index]}')
print(f'Cross-elasticity at top end of distance: {cross_elasticity[top_index]}')

# Get the vector of distances
#empirical_distances = np.array(list(movie_distances.values()))

# Plot data density and cross-elasticity on same plot
fig, ax1 = plt.subplots()

# Add title
ax1.set_title('Cross-Elasticity of Distance')
ax2 = ax1.twinx()

# Plot the empirical distances
ax2.hist(empirical_distances, bins=20, alpha=0.5, density=True)
ax2.set_xlabel('Distance')
ax2.set_ylabel('Density of Empirical Distance')

ax1.plot(distances, cross_elasticity, color='orange')
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

ax1.plot(distances, cross_elasticity, color='orange')
ax1.set_ylabel('Cross-Elasticity of Distance')

ax2 = ax1.twinx()

# Plot the empirical distances
ax2.hist(empirical_distances, bins=20, alpha=0.5, density=True)
ax2.set_xlabel('Distance')
ax2.set_ylabel('Density of Empirical Distance')

# Add vertical lines at the top and bottom ends
ax2.axvline(x=top_end, color='red', linestyle='--')
ax2.axvline(x=bottom_end, color='red', linestyle='--')

fig.tight_layout()
plt.savefig('output/cross_elasticity_zoom.png')
