import pandas as pd
import numpy as np
from scipy.optimize import minimize
from optimparallel import minimize_parallel
import matplotlib.pyplot as plt
import linearmodels as lm
from stargazer.stargazer import Stargazer
import warnings
import cProfile
import pstats

# Turn off un-actionable FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# For weeks greater than this, we truncate to this
# Note zero-indexing; thus, this is actually the *fifth* week
WEEK_THRESHOLD = 4

# Load the embeddings from parquet
tmdb = pd.read_parquet('../tmdb/embeddings/movie_descriptions.parquet')

# Drop if embeddings are null
tmdb = tmdb.dropna(subset=['embedding'])
tmdb = tmdb.reset_index(drop=True)

# Filter to primary genre
tmdb['tmdb_genre'] = tmdb['tmdb_genres'].apply(lambda x: x[0] if len(x) > 0 else None)

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

# Replace genre with label
tmdb['tmdb_genre'] = tmdb['tmdb_genre'].map(genre_labels)

# Create bar chart of genres
genre_counts = tmdb['tmdb_genre'].value_counts()
genre_counts.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of Movies by Genre')

# Label the genres
plt.xticks(ticks=range(len(genre_counts)), rotation=45)

plt.savefig('output/genre_counts.png')

# Clear plot
plt.clf()

# Get array of movie ids
movie_ids = tmdb['movie_id'].copy()

# Create a map from movie_ids to indices
movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

raw_embeddings = np.array(tmdb['embedding'].values)
raw_embeddings = np.array([np.array(x) for x in raw_embeddings])

# Create distance matrix by taking dot product of embeddings
distances = 1 - raw_embeddings @ raw_embeddings.T

# Normalize distances
distances = distances / np.max(distances)

# Load box office info
guru = pd.read_parquet('../guru/data/weekly_sales.parquet')

guru = guru.reset_index(drop=False)

# Merge on movie genre
guru = guru.merge(tmdb[['movie_id', 'tmdb_genre']], on='movie_id', how='left')

# Create dummy for whether another movie of that genre is showing within each week
# guru['genre_clash'] = guru.groupby(['Date', 'tmdb_genre'])['movie_id'].transform('count') > 1

# Compute log interest
guru['ln_earnings'] = np.log(guru['Week Sales'])

# Drop if missing weeks
guru = guru.dropna(subset=['Weeks'])

print(guru['Weeks'].describe())

# Create histogram of weeks since release
plt.hist(guru['Weeks'], bins=20)
plt.xlabel('Weeks Since Release')
plt.ylabel('Count')
plt.title('Histogram of Weeks Since Release')
plt.savefig('output/weeks_histogram.png')

# We want their weeks to also be zero-indexed so we can use it as an index
guru['Weeks'] = guru['Weeks'] - 1

# Truncate weeks to limit degrees of freedom
guru['Weeks'] = np.minimum(guru['Weeks'], WEEK_THRESHOLD)

# Convert date to datetime
guru['Date'] = pd.to_datetime(guru['Date'])

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


# Per Nick: could include lambda in the regression and target that those coefficients are 1
# For each date, pre-compute the values for each movie pair
dates = guru['Date'].unique()
date_movie_dict = {}
for date in dates:
    date_movies = guru[guru['Date'] == date]
    date_movie_dict[date] = date_movies

def regress_given_lambda(age_coefficients, guru):

    guru = guru.copy()

    # For each movie-date, compute the leave-one-out set of distances within the date
    gamma0 = []
    gamma1 = []
    gamma2 = []
    gamma3 = []

    for row in guru.itertuples():

        # Can I avoid redundant computation within a date?
        movie_id = row.movie_id
        date = row.Date

        # Get index of movie_id in movie_ids
        movie_index = movie_id_to_index[movie_id]

        # Get all movies on the date
        date_movies = date_movie_dict[date]

        # Can ignore that self is included since distance(x, x) = 0
        competitor_age_coefs = age_coefficients[date_movies['Weeks']]
        competitor_indices = np.array([movie_id_to_index[movie_id] for movie_id in date_movies['movie_id']])
        competitor_distances = distances[movie_index, competitor_indices]

        # No need to weight by log price since log price is constant
        # If using weighted distances, subtract off one for self
        # Don't want to raise age coefficient to power when computing distances
        gamma0.append(len(competitor_distances) - 1)
        gamma1.append(np.sum(competitor_age_coefs * competitor_distances))
        gamma2.append(np.sum(competitor_age_coefs * (competitor_distances ** 2)))
        gamma3.append(np.sum(competitor_age_coefs * (competitor_distances ** 3)))

    # Add to the guru data
    guru['gamma0'] = gamma0
    guru['gamma1'] = gamma1
    guru['gamma2'] = gamma2
    guru['gamma3'] = gamma3

    # Convert movie_id to categorical
    guru['movie_id'] = guru['movie_id'].astype('category')

    guru.set_index(['movie_id', 'Date'], inplace=True)

    # Run regression of log earnings on distances and age
    # NOTE: WITHIN A WEEKEND, THERE IS NO VARIATION IN THE # OF MOVIES. THUS, GAMMA0 GETS OMITTED
    formula = 'ln_earnings ~ gamma1 + gamma2 + gamma3 + C(Weeks) + EntityEffects + TimeEffects'
    reg = lm.PanelOLS.from_formula(formula=formula, data=guru, drop_absorbed=True)

    results = reg.fit(low_memory=False)

    print(f'Lambda: {age_coefficients}')
    print(f'RMSE: {results.model_ss}')

    return results

def compute_model_error(age_coefficients, guru):

    # Run the regression
    model = regress_given_lambda(age_coefficients, guru)

    return model.model_ss


# Profile the objective function
#cProfile.run('regress_given_lambda(np.ones(WEEK_THRESHOLD + 1), guru)', 'output/profile.prof')
#p = pstats.Stats('output/profile.prof')
#p.strip_dirs().sort_stats('cumulative').print_stats(10)

# Optimize the age coefficients
print("Minimizing...")
initial_guess = np.ones(WEEK_THRESHOLD + 1)
#minimization = minimize_parallel(
minimization = minimize(
    fun = compute_model_error, 
    x0 = initial_guess,
    args = (guru,),
)

# Get the optimal age coefficients
age_coefficients = minimization.x
optimal_model = regress_given_lambda(age_coefficients, guru)

print(optimal_model)

stargazer = Stargazer([optimal_model])
stargazer.covariate_order(['gamma0', 'gamma1', 'gamma2', 'gamma3'])
stargazer.add_custom_notes([
    "Omitting movie age, movie, and date fixed effects for brevity"
])
latex = stargazer.render_latex(
    escape=True,
    only_tabular=True,
)
with open('output/cosine_regression.tex', 'w') as f:
    f.write(latex)

# Get coefficients on gamma1, gamma2, and gamma3
gamma0_coefficient = optimal_model.params['gamma0']
gamma1_coefficient = optimal_model.params['gamma1']
gamma2_coefficient = optimal_model.params['gamma2']
gamma3_coefficient = optimal_model.params['gamma3']

# Plot cross-elasticities over distance
distance_grid = np.linspace(0, 1, 100)
cross_elasticity = gamma0_coefficient + gamma1_coefficient * distance_grid + gamma2_coefficient * distance_grid ** 2 + gamma3_coefficient * distance_grid ** 3

bottom_index = np.argmin(np.abs(distance_grid - bottom_end))
top_index = np.argmin(np.abs(distance_grid - top_end))


print(f'Cross-elasticity at bottom end of distance: {cross_elasticity[bottom_index]}')
print(f'Cross-elasticity at top end of distance: {cross_elasticity[top_index]}')

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
#ymin = np.min(cross_elasticity[bottom_index:top_index])
#ymax = np.max(cross_elasticity[bottom_index:top_index])
#margin = (ymax - ymin)
#ax1.set_ylim(ymin - margin, ymax + margin)
ax1.set_ylim(0.50, 0.55)

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