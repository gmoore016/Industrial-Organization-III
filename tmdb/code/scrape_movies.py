import tmdbsimple as tmdb
import csv

TARANTINO = [466272, 24, 16869, 68718, 273248]
PIXAR = [862, 585, 12, 9806, 10681]
TWINS = [
    50544, # No Strings Attached
    41630, # Friends with Benefits
    20352, # Despicable Me
    38055, # Megamind
    374720, # Dunkirk
    399404, # Darkest Hour
]

# Read API key from file
with open('tmdb.secret', 'r') as f:
    api_key = f.read().strip()
    tmdb.API_KEY = api_key

# Get movie descriptions
def get_movie_descriptions(movie_ids):
    descriptions = []
    for movie_id in movie_ids:
        movie = tmdb.Movies(movie_id)
        response = movie.info()
        descriptions.append(response['overview'])
    return descriptions

# Get descriptions for Tarantino movies
tarantino_descriptions = get_movie_descriptions(TARANTINO)

# Write descriptions to file
with open('descriptions/tarantino.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    for i, description in enumerate(tarantino_descriptions):
        writer.writerow([TARANTINO[i], description, "tarantino"])

# Get descriptions for Pixar movies
pixar_descriptions = get_movie_descriptions(PIXAR)

# Write descriptions to file
with open('descriptions/pixar.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    for i, description in enumerate(pixar_descriptions):
        writer.writerow([PIXAR[i], description, "pixar"])

# Get descriptions for twin movies
twin_descriptions = get_movie_descriptions(TWINS)

# Write descriptions to file
with open('descriptions/twins.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'description', 'group'])
    counter = 0
    for i, description in enumerate(twin_descriptions):
        writer.writerow([TWINS[i], description, "twin " + str(counter // 2)])
        counter += 1