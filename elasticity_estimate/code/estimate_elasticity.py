import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the embeddings
embeddings = pickle.load(open('../tmdb/embeddings/movie_descriptions.pkl', 'rb'))

# Rename movie_odid to movie_id
embeddings.rename(columns={'movie_odid': 'movie_id'}, inplace=True)

# Merge embedding data onto google tren

raw_embeddings = embeddings['embedding'].apply(lambda x: x.data[0].embedding).to_list()

raw_embeddings = np.array(raw_embeddings)
reduced_embeddings = PCA(n_components=6).fit_transform(raw_embeddings)

for i in range(6):
    embeddings[f'pca_{i}'] = reduced_embeddings[:, i]

# Load google trends
google_trends = pd.read_csv('../google_trends/output/trends.csv')

# Filter only to dates with 100 interest
premier_dates = google_trends[google_trends['interest'] == 100]

# In the event of a tie, take the first date
premier_dates = premier_dates.groupby('movie_id').first()
premier_dates.reset_index(inplace=True)

# Rename columns
premier_dates.columns = ['movie_id', 'premier_date', 'interest']
premier_dates = premier_dates[['movie_id', 'premier_date']]

# Merge premier_dates onto trends
google_trends = google_trends.merge(premier_dates, on='movie_id')

# Keep if date is within three months of premier
google_trends['date'] = pd.to_datetime(google_trends['date'])
google_trends['premier_date'] = pd.to_datetime(google_trends['premier_date'])
google_trends['days_from_premier'] = (google_trends['date'] - google_trends['premier_date']).dt.days
google_trends = google_trends[google_trends['days_from_premier'].between(0, 90)]

# Keep if date has positive interest
google_trends = google_trends[google_trends['interest'] > 0]

# Limit only to movies from 2007
google_trends = google_trends[google_trends['premier_date'].dt.year == 2007]

# Compute log interest
google_trends['ln_interest'] = np.log(google_trends['interest'])

# De-Mean log interest within a month
google_trends['month'] = google_trends['date'].dt.to_period('M')
google_trends['ln_interest'] = google_trends.groupby('month')['ln_interest'].transform(lambda x: x - x.mean())

# De-Mean log interest within a movie
google_trends['ln_interest'] = google_trends.groupby('movie_id')['ln_interest'].transform(lambda x: x - x.mean())

# Limit sample to movies with trends and embeddings
movies_with_trends = google_trends['movie_id'].unique()
movies_with_embeddings = embeddings['movie_id'].unique()
movies = np.intersect1d(movies_with_trends, movies_with_embeddings)



google_trends = google_trends[google_trends['movie_id'].isin(movies)]
embeddings = embeddings[embeddings['movie_id'].isin(movies)]

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

plt.hist(empirical_distances, bins=20)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Distances')
plt.show()

# Plot elasticity function
plt.plot(distance_grid, elasticity)
plt.xlabel('Distance')
plt.ylabel('Elasticity')
plt.title('Estimated Elasticity Function')

# Get bottom and top decile
distances = np.array(empirical_distances)
bottom_decile = np.percentile(distances, 10)
top_decile = np.percentile(distances, 90)
median = np.percentile(distances, 50)

# Add lines for median and top/bottom deciles
plt.axvline(x=bottom_decile, color='r', linestyle='--')
plt.axvline(x=top_decile, color='r', linestyle='--')
plt.axvline(x=median, color='g', linestyle='--')

# Add labels
plt.text(bottom_decile, 0, 'Bottom Decile', rotation=90)
plt.text(top_decile, 0, 'Top Decile', rotation=90)
plt.text(median, 0, 'Median', rotation=90)

plt.show()

