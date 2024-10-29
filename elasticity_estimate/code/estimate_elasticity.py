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
WEEK_THRESHOLD = 9

# Initial guess for gamma
INITIAL_GUESS = np.array([1, .1, .01, .001])


def get_neighbors(example, distances, tmdb, movie_ids):
    '''
    Function to get characteristic examples for a given movie
    '''
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

def compute_model_parameters(guru, movie_id_to_index, date_movie_dict, distances):
        """
        Computes optimal model parameters through two-stage minimization process.
        First finds approximate parameters on small sample, then refines on full sample.
        
        Args:
            guru (pd.DataFrame): Full dataset
            movie_id_to_index (dict): Mapping of movie IDs to indices
            date_movie_dict (dict): Pre-computed movie data by date
            distances (np.array): Distance matrix between movies
            
        Returns:
            tuple: Optimal parameters (gamma) and fitted model
        """
        print("Minimizing...")

        # For initial search, limit only to one year
        sample_guru = guru[guru['Date'].dt.year == 2000]

        # The minimiziation process is expensive; thus, 
        # we get "close" to the optimum with a small sample, then
        # use that as the initial guess for the full sample
        initial_minimization = minimize(
            fun = compute_model_error, 
            x0 = INITIAL_GUESS,
            args = (sample_guru, movie_id_to_index, date_movie_dict, distances),
        )

        # Get the optimal age coefficients
        gamma_star = initial_minimization.x

        final_minimization = minimize(
            fun = compute_model_error,
            x0 = gamma_star,
            args = (guru, movie_id_to_index, date_movie_dict, distances),
        )

        gamma_star = final_minimization.x

        print("Initial Minimiziation Complete!")

        optimal_model = regress_given_gamma(gamma_star, guru, movie_id_to_index, date_movie_dict, distances)
        
        return gamma_star, optimal_model


def regress_given_gamma(gamma, guru, movie_id_to_index, date_movie_dict, distances):
    """Function for nonlinear optimization of age coefficients"""

    guru = guru.copy()

    # Define cubic dependant on gamma
    cubic = lambda x: gamma[0] + gamma[1] * x + gamma[2] * x ** 2 + gamma[3] * x ** 3

    rows = []

    for row in guru.itertuples():
        # Get the movie's date and id
        movie_id = row.movie_id
        date = row.Date
        age = row.Weeks
        ln_earnings = row.ln_earnings

        # Get index of movie_id in movie_ids
        movie_index = movie_id_to_index[movie_id]

        # Create a lambda vector with length equal to the number of weeks
        lambda_vals = np.zeros(WEEK_THRESHOLD)
        lambda_vals[age] = 1

        # Create an alpha vector with length equal to the number of movies
        alpha_vals = np.zeros(len(movie_id_to_index))
        alpha_vals[movie_index] = 1

        # Get all movies on the date not equal to the original movie
        date_movies = date_movie_dict[date]
        date_movies = date_movies[date_movies['movie_id'] != movie_id]

        competitor_ids = date_movies['movie_id'].values
        competitor_ages = date_movies['Weeks'].values
        competitor_indices = np.array([movie_id_to_index[competitor_id] for competitor_id in competitor_ids])

        # CAN I PRE-COMPUTE THIS ONCE AT THE START OF THE FUNCTION, RATHER THAN RE-DOING IT FOR EACH ROW?
        fs_of_distances = cubic(distances[movie_index, competitor_indices])
        
        alpha_vals[competitor_indices] += fs_of_distances
        lambda_vals[competitor_ages] += fs_of_distances

        row = np.concatenate((alpha_vals, lambda_vals, [date, ln_earnings, movie_id]))
        rows.append(row)


    # Create a dataframe of the alpha and lambda values along with the date
    regression_df = pd.DataFrame(rows)

    # Rename columns
    regression_df.columns = [f'alpha_{index}' for index in range(len(movie_id_to_index))] + [f'lambda_{i}' for i in range(WEEK_THRESHOLD)] + ['date', 'ln_earnings', 'movie_id']

    # Set the index to the date and movie_id
    regression_df['date'] = pd.to_datetime(regression_df['date'])
    regression_df = regression_df.set_index(['movie_id', 'date'])

    # Drop any columns with all zeros
    # Should only be relevant in heuristic subsample
    regression_df = regression_df.loc[:, (regression_df != 0).any(axis=0)]

    # Get all remaining alpha and lambda col names
    alpha_cols = [colname for colname in regression_df.columns if 'alpha' in colname]
    lambda_cols = [colname for colname in regression_df.columns if 'lambda' in colname]

    # Define the regression formula
    reg_formula = 'ln_earnings ~ ' + ' + '.join(alpha_cols) + ' + ' + ' + '.join(lambda_cols) + ' + TimeEffects'

    reg = lm.PanelOLS.from_formula(formula=reg_formula, data=regression_df, drop_absorbed=True)

    results = reg.fit(low_memory=False)

    return results

def compute_model_error(gamma, guru, movie_id_to_index, date_movie_dict, distances):
    """Helper function which returns only the model sum of squares"""

    # Run the regression
    model = regress_given_gamma(gamma, guru, movie_id_to_index, date_movie_dict, distances)

    residuals = model.resids
    rmse = np.sqrt(np.mean(residuals ** 2))

    return rmse


def main():
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

    plt.savefig('output/genre_counts.png', bbox_inches='tight')

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

    # Compute log interest
    guru['ln_earnings'] = np.log(guru['Week Sales'])

    # Convert date to datetime
    guru['Date'] = pd.to_datetime(guru['Date'])

    # Keep only movies in 2019 or earlier
    guru = guru[(2000 <= guru['Date'].dt.year) & (guru['Date'].dt.year <= 2019)]

    # Drop if missing weeks
    guru = guru.dropna(subset=['Weeks'])

    print(guru['Weeks'].describe())

    # Create histogram of weeks since release
    plt.hist(guru['Weeks'], bins=20)
    plt.xlabel('Weeks Since Release')
    plt.ylabel('Count')
    plt.title('Histogram of Weeks Since Release')
    plt.savefig('output/weeks_histogram.png', bbox_inches='tight')

    # Get the movie with the longest run
    longest_run = guru[guru['Weeks'] == guru['Weeks'].max()]
    print(longest_run)

    # We want their weeks to also be zero-indexed so we can use it as an index
    guru['Weeks'] = guru['Weeks'] - 1

    # Truncate weeks to limit degrees of freedom
    guru['Weeks'] = np.minimum(guru['Weeks'], WEEK_THRESHOLD - 1)



    # Limit sample to movies with trends and embeddings
    movies_with_trends = guru['movie_id'].unique()
    movies_with_embeddings = tmdb['movie_id'].unique()
    movies = np.intersect1d(movies_with_trends, movies_with_embeddings)

    # No longer true since we're adding a heuristic subsample
    #assert len(movies) == distances.shape[0]

    print(f'Number of movies satisfying all criteria: {len(movies)}')

    guru = guru[guru['movie_id'].isin(movies)]

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
    #for example in examples:
    #    get_neighbors(example, distances, tmdb, movie_ids)


    # For each date, pre-compute the values for each movie pair
    dates = guru['Date'].unique()
    date_movie_dict = {}
    for date in dates:
        date_movies = guru[guru['Date'] == date]
        date_movie_dict[date] = date_movies

    print(f"Number of dates: {len(dates)}")

    # Profile the objective function
    #cProfile.run('compute_model_error(INITIAL_GUESS, guru, movie_id_to_index, date_movie_dict, distances)', 'output/profile.prof')
    #p = pstats.Stats('output/profile.prof')
    #p.strip_dirs().sort_stats('cumulative').print_stats(10)

    # Compute the model parameters using the true data
    gamma_star, optimal_model = compute_model_parameters(guru, movie_id_to_index, date_movie_dict, distances)

    # Bootstrap to get confidence intervals
    print("\nBootstrapping to get confidence intervals...")
    
    n_bootstrap = 2
    bootstrap_gammas = []
    bootstrap_models = []
    # Get unique dates for sampling
    unique_dates = pd.Series(list(date_movie_dict.keys()))
    
    for i in range(n_bootstrap):
        print(f"\nBootstrap iteration {i+1}/{n_bootstrap}")
        
        # Sample dates with replacement
        sampled_dates = unique_dates.sample(n=len(unique_dates), replace=True)
        
        # Get the bootstrap sample
        bootstrap_guru = guru[guru['Date'].isin(sampled_dates)]
        
        # Compute parameters on bootstrap sample
        try:
            bootstrap_gamma, bootstrap_model = compute_model_parameters(
                bootstrap_guru, 
                movie_id_to_index,
                date_movie_dict,
                distances
            )
            bootstrap_gammas.append(bootstrap_gamma)
            bootstrap_models.append(bootstrap_model)
        except:
            print(f"Failed on bootstrap iteration {i+1}")
            continue

    # Plot results
    distance_grid = np.linspace(0, 1, 100)

    bootstrap_cross_elasticities = []
    for bootstrap_gamma in bootstrap_gammas:
        bootstrap_cubic = lambda x: bootstrap_gamma[0] + bootstrap_gamma[1] * x + bootstrap_gamma[2] * x ** 2 + bootstrap_gamma[3] * x ** 3

        bootstrap_cross_elasticity = bootstrap_cubic(distance_grid)
        bootstrap_cross_elasticities.append(bootstrap_cross_elasticity)

    # For each point on the grid, compute the 95% CI
    bootstrap_cross_elasticities = np.array(bootstrap_cross_elasticities)
    lower_bounds = np.percentile(bootstrap_cross_elasticities, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrap_cross_elasticities, 97.5, axis=0)

    # Plot cross-elasticities over distance

    cross_elasticity = gamma_star[0] + gamma_star[1] * distance_grid + gamma_star[2] * distance_grid ** 2 + gamma_star[3] * distance_grid ** 3

    bottom_index = np.argmin(np.abs(distance_grid - bottom_end))
    top_index = np.argmin(np.abs(distance_grid - top_end))

    # Flatten the distances
    distances = distances.flatten()

    # Plot data density and cross-elasticity on same plot
    fig, ax1 = plt.subplots()

    # Add title
    ax1.set_title('Impact over Distance')
    ax1.set_xlabel('Distance')
    ax2 = ax1.twinx()

    # Plot the empirical distances
    ax2.hist(distances, bins=20, alpha=0.5, density=True)
    ax2.set_ylabel('Density of Empirical Distance')

    ax1.plot(distance_grid, cross_elasticity, color='orange')
    ax1.set_ylabel('Competition Function')

    ax1.fill_between(distance_grid, lower_bounds, upper_bounds, color='blue', alpha=0.2)

    ax2.axvline(x=top_end, color='red', linestyle='--')
    ax2.axvline(x=bottom_end, color='red', linestyle='--')

    fig.tight_layout()
    plt.savefig('output/gamma.png', bbox_inches='tight')

    # Zoom in on the cross-elasticity in the support of the data

    fig, ax1 = plt.subplots()

    # Limit only to bottom 10% to top 10%
    ax1.set_xlim(bottom_end - .2, top_end + .2)

    # Bound the y axis based on range between bottom and top end
    #ymin = np.min(cross_elasticity[bottom_index:top_index])
    #ymax = np.max(cross_elasticity[bottom_index:top_index])
    #margin = (ymax - ymin)
    #ax1.set_ylim(ymin - margin, ymax + margin)
  
    # Get the max observed within the bounds
    max_observed = np.max(cross_elasticity[bottom_index:top_index])
    min_observed = np.min(cross_elasticity[bottom_index:top_index])
    margin = (max_observed - min_observed) / 2
    ax1.set_ylim(min_observed - margin, max_observed + margin)

    # Add a dotted horizontal line at zero
    ax1.axhline(y=0, color='grey', linestyle='--')

    # Add title
    ax1.set_title('Impact over Distance')

    ax1.plot(distance_grid, cross_elasticity, color='orange')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Competition Function')

    ax2 = ax1.twinx()

    # Plot the empirical distances
    ax2.hist(distances, bins=20, alpha=0.5, density=True)
    ax2.set_ylabel('Density of Empirical Distance')

    # Add vertical lines at the top and bottom ends
    ax2.axvline(x=top_end, color='red', linestyle='--')
    ax2.axvline(x=bottom_end, color='red', linestyle='--')

    fig.tight_layout()
    plt.savefig('output/gamma_zoomed.png', bbox_inches='tight')

    # Let's try a stacked plot rather than twin axes
    plt.clf()
    fig, axs = plt.subplots(2, sharex=True)

    # Set the x limits
    axs[0].set_xlim(bottom_end - .2, top_end + .2)
    axs[1].set_xlim(bottom_end - .2, top_end + .2)

    # Plot the cross-elasticity
    axs[0].plot(distance_grid, cross_elasticity)
    axs[0].axhline(y=0, color='grey', linestyle='--')
    axs[0].set_ylabel('Competition Function')
    axs[0].set_ylim(min_observed - margin, max_observed + margin)

    # Plot the empirical distances
    axs[1].hist(distances, bins=20, density=True)
    axs[1].set_ylabel('Density of Data')
    axs[1].set_xlabel('Distance')

    # Add vertical lines at the top and bottom ends
    axs[0].axvline(x=top_end, color='red', linestyle='--')
    axs[0].axvline(x=bottom_end, color='red', linestyle='--')
    axs[1].axvline(x=top_end, color='red', linestyle='--')
    axs[1].axvline(x=bottom_end, color='red', linestyle='--')

    fig.tight_layout()
    plt.savefig('output/gamma_zoom_stacked.png', bbox_inches='tight')




    # Get the age coefficients and errors
    lambda_cols = [colname for colname in optimal_model.params.index if 'lambda' in colname]
    lambda_values = optimal_model.params[lambda_cols]
    lambda_errors = optimal_model.std_errors[lambda_cols]

    # Plot the age coefficients and errors
    plt.clf()
    plt.errorbar(range(WEEK_THRESHOLD), lambda_values, yerr=lambda_errors, fmt='o')
    plt.xlabel('Weeks Since Release')
    plt.ylabel('Fixed Effect')
    plt.title('Age Fixed Effects')
    plt.savefig('output/lambda_coefficients.png', bbox_inches='tight')


    # Get the alpha coefficients
    alpha_cols = [colname for colname in optimal_model.params.index if 'alpha' in colname]
    alpha_values = optimal_model.params[alpha_cols]

    # Plot a distribution of the alpha coefficients
    plt.clf()
    plt.hist(alpha_values, bins=20)

    # Add labels
    plt.xlabel('Alpha Coefficient')
    plt.ylabel('Count')
    plt.title('Distribution of Alpha Coefficients')

    # Add a vertical line at the mean
    plt.axvline(x=alpha_values.mean(), color='red', linestyle='--')
    # Print the mean
    #plt.text(alpha_values.mean(), 100, f'Mean: {alpha_values.mean():.2f}', rotation=90)
    plt.text(4, 250, f'Mean: {alpha_values.mean():.2f}')
    plt.text(4, 235, f'SD: {alpha_values.std():.2f}')

    plt.savefig('output/alpha_coefficients.png', bbox_inches='tight')




if __name__ == '__main__':
    main()