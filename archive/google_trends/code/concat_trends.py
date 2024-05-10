import pandas as pd 
import os 
import matplotlib.pyplot as plt

# Load each of the files pulled from google trends
trends_files = os.listdir('raw')

# Load each of the files into a dataframe
trends_data = []
for file in trends_files:
    data = pd.read_csv(f'raw/{file}')
    trends_data.append(data)

# Concatenate all the dataframes
trends_data = pd.concat(trends_data)
trends_data['date'] = pd.to_datetime(trends_data['date'])

# Set multiindex of movie id and date
trends_data.set_index(['movie_id', 'date'], inplace=True)

new_index = pd.MultiIndex.from_product(trends_data.index.levels)
trends_data = trends_data.reindex(new_index)

# Fill in missing values with 0
trends_data.fillna(0, inplace=True)

trends_data.reset_index(inplace=True)

# Filter to the Dark Knight
dark_knight = trends_data[trends_data['movie_id'] == 20100]

# Plot the data
# Show data labels yearly on the x axis
plt.plot(
    dark_knight['date'], 
    dark_knight['interest'],
)
plt.xticks(
    dark_knight['date'][::52],
    [str(date.year) for date in dark_knight['date'][::52]],
    rotation=45,
)

plt.xlabel('Date')
plt.ylabel('Interest')
plt.title('Interest in The Dark Knight')

# Create datetime object for july 18, 2008
release_date = pd.to_datetime('2008-07-18')
plt.axvline(release_date, color='red', linestyle='--')

#plt.show()
plt.savefig('output/dark_knight.png')

# Save the data
trends_data.to_csv(
    'output/trends.csv',
    index=False,
)