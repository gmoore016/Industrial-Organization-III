import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re

# Load the embeddings
embeddings = pickle.load(open('../tmdb/embeddings/movie_descriptions.pkl', 'rb'))

# Rename movie_odid to movie_id
embeddings.rename(columns={'tmdb_id': 'movie_id'}, inplace=True)

raw_embeddings = embeddings['embedding'].apply(lambda x: x.data[0].embedding).to_list()

raw_embeddings = np.array(raw_embeddings)
reduced_embeddings = PCA(n_components=6).fit_transform(raw_embeddings)

for i in range(6):
    embeddings[f'pca_{i}'] = reduced_embeddings[:, i]

# Load box office info
guru = pd.read_parquet('../guru/data/weekly_sales.parquet')

guru = guru.reset_index(drop=False)

# Drop if missing theaters
guru = guru.dropna(subset=['Theaters'])

# Convert theaters and weeks to integers
guru['Theaters'] = guru['Theaters'].astype(pd.Int64Dtype())
guru['Weeks'] = guru['Weeks'].astype(pd.Int64Dtype())


# Compute log interest
guru['ln_earnings'] = np.log(guru['Week Sales'])

# Count if missing theaters or log earnings
print(f"Missing theaters: {guru['Theaters'].isnull().sum()}")
print(f"Missing log earnings: {guru['ln_earnings'].isnull().sum()}")

guru = guru.dropna(subset=['Theaters', 'ln_earnings'])
guru = guru.drop("Weeks", axis=1)

# Disambiguate movies with the same name while computing movie spell lengths
max_spell_length = 10000
counter = 0
while max_spell_length > 60:
    print("Iteration: ", counter)
    wide_release_dates = guru[guru['Theaters'] >= 600].groupby('Clean Title')['Date'].min()
    # Rename the column
    wide_release_dates = wide_release_dates.reset_index()
    wide_release_dates.columns = ['Clean Title', 'Date_wide']

    # Drop any showings before the wide release
    guru = guru.merge(
        wide_release_dates, 
        on='Clean Title',
    )
    
    guru = guru[guru['Date'] >= guru['Date_wide']]

    # Compute weeks since wide release
    guru['weeks_since_release'] = (guru['Date'] - guru['Date_wide']).dt.days // 7

    # If a movie has been out for more than 50 weeks, it's likely a different movie with the same name
    # Thus, flag title with counter
    guru['Clean Title'] = np.where(guru['weeks_since_release'] > 60, guru['Clean Title'] + f'_{counter}', guru['Clean Title'])

    max_spell_length = guru['weeks_since_release'].max()
    counter += 1
    
    # Drop wide release date
    print(guru.head())
    print(guru.columns)
    guru = guru.drop('Date_wide', axis=1)

# Histogram of weeks since release
plt.hist(guru['weeks_since_release'], bins=20)
plt.xlabel('Weeks Since Wide Release')
plt.ylabel('Frequency')
plt.title('Histogram of Weeks Since Wide Release')
plt.show()

spell_lengths = guru.groupby('Clean Title')['weeks_since_release'].max()
print(spell_lengths.sort_values(ascending=False))
kill



# De-Mean log interest within a month
guru['month'] = guru['Date'].dt.to_period('M')
guru['ln_earnings'] = guru.groupby('month')['ln_earnings'].transform(lambda x: x - x.mean())

# De-Mean log interest within a movie
guru['ln_earnings'] = guru.groupby('movie_id')['ln_earnings'].transform(lambda x: x - x.mean())



# Limit sample to movies with trends and embeddings
movies_with_trends = guru['Clean Title'].unique()
movies_with_embeddings = embeddings['Clean Title'].unique()
movies = np.intersect1d(movies_with_trends, movies_with_embeddings)

print(f'Number of movies satisfying all criteria: {len(movies)}')

guru = guru[guru['Clean Title'].isin(movies)]
embeddings = embeddings[embeddings['Clean Title'].isin(movies)]

# Create distances between relevant movies
movie_distances = {}

max_distance = 0

for movie_id1 in movies:
    for movie_id2 in movies:
        if movie_id1 == movie_id2:
            continue
        if (movie_id2, movie_id1) in movie_distances:
            movie_distances[(movie_id1, movie_id2)] = movie_distances[(movie_id2, movie_id1)]

        movie1 = embeddings[embeddings['movie_id'] == movie_id1]
        movie2 = embeddings[embeddings['movie_id'] == movie_id2]

        # Filter only to pca columns
        movie1 = movie1.filter(like='pca').to_numpy().flatten()
        movie2 = movie2.filter(like='pca').to_numpy().flatten()

        distance = np.linalg.norm(movie1 - movie2)
        if distance > max_distance:
            max_distance = distance
        movie_distances[(movie_id1, movie_id2)] = distance

# Normalize distances
for key, value in movie_distances.items():
    movie_distances[key] = value / max_distance

# Iterate over weeks of the dataframe
weeks = google_trends['date'].unique()
movie_week_distances = {}
for week in weeks:
    week_data = google_trends[google_trends['date'] == week]

    # For each movie, compute the list of distances
    week_movies = week_data['movie_id'].unique()
    for base_movie in week_movies:
        distances = []
        for comparison_movie in week_movies:
            if base_movie == comparison_movie:
                continue
            distances.append(movie_distances[(base_movie, comparison_movie)])

        movie_week_distances[(base_movie, week)] = distances

class Observation():
    def __init__(self, movie_id, date, distances, ln_interest):
        self.movie_id = movie_id
        self.date = date
        self.distances = distances
        self.ln_interest = ln_interest

# Create observations
observations = []
for (movie_id, date), distances in movie_week_distances.items():
    interest = google_trends[(google_trends['movie_id'] == movie_id) & (google_trends['date'] == date)]['ln_interest'].values[0]
    observation = Observation(movie_id, date, distances, interest)
    observations.append(observation)

def objective(params, observations):
    """
    Objective function for the optimization
    """
    # Unpack parameters
    alpha, gamma1, gamma2, gamma3 = params

    # Compute the sum of squared errors
    sse = 0
    for observation in observations:
        # Cross-substitution term
        cross_sub = 0
        for distance in observation.distances:
            cross_sub += gamma1 * distance + gamma2 * distance ** 2 + gamma3 * distance ** 3

        interest_hat = alpha + cross_sub
        sse += (observation.ln_interest - interest_hat) ** 2

    return sse

# Estimate parameters for our cubic
solution = minimize(
    objective,
    x0=[1, 1, 1, 1],
    args=(observations,),
)

print(solution.x)

# Plot function implied by solution
alphahat, gamma1hat, gamma2hat, gamma3hat = solution.x

distance_grid = np.linspace(0, 1, 100)
elasticity = gamma1hat * distance_grid + gamma2hat * distance_grid ** 2 + gamma3hat * distance_grid ** 3


# Create histogram of distances
empirical_distances = []
for observation in observations:
    empirical_distances.extend(observation.distances)

# Get bottom and top decile
distances = np.array(empirical_distances)
bottom_decile = np.percentile(distances, 10)
top_decile = np.percentile(distances, 90)
median = np.percentile(distances, 50)

plt.hist(empirical_distances, bins=20)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Distances')

plt.axvline(x=bottom_decile, color='r', linestyle='--')
plt.axvline(x=top_decile, color='r', linestyle='--')
plt.axvline(x=median, color='g', linestyle='--')

# Add labels
plt.text(bottom_decile, 0, 'Bottom Decile', rotation=90)
plt.text(top_decile, 0, 'Top Decile', rotation=90)
plt.text(median, 0, 'Median', rotation=90)

plt.savefig('output/histogram_distances.png')


# New plot
plt.figure()

# Plot elasticity function
plt.plot(distance_grid, elasticity)
plt.xlabel('Distance')
plt.ylabel('Elasticity')
plt.title('Estimated Elasticity Function')

# Add lines for median and top/bottom deciles
plt.axvline(x=bottom_decile, color='r', linestyle='--')
plt.axvline(x=top_decile, color='r', linestyle='--')
plt.axvline(x=median, color='g', linestyle='--')

# Add labels
plt.text(bottom_decile, 0, 'Bottom Decile', rotation=90)
plt.text(top_decile, 0, 'Top Decile', rotation=90)
plt.text(median, 0, 'Median', rotation=90)

plt.savefig('output/elasticity_function.png')

