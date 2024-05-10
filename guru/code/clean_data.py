import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import progressbar
import string
from unidecode import unidecode
import numpy as np

PAREN_REGEX = re.compile(r'\([^)]*\)')
table_regex = re.compile(r"Top\s*5")

def parse_html(filename):
    with open(f"html/{filename}") as file:
        html = file.read()

    # Remove all newlines
    html = html.replace("\n", " ")

    # Parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Get the list of table elements and select the last one
    tables = soup.find_all("table")
    if len(tables) == 0:
        raise ValueError("No Tables")
    table = tables[-1]


    # Convert the table to a DataFrame
    df = pd.read_html(
        str(table),
        match=table_regex,
    )
    #assert len(df) == 1
    df = df[0]

    converters = {c: lambda x: str(x) for c in df.columns}
    df = pd.read_html(
        str(table),
        match=table_regex,
        converters=converters
    )[0]

    # Replace string nan with actual NaN
    df = df.replace("nan", pd.NA)

    # Drop empty rows at the top
    while pd.isna(df[1][0]):
        df = df[1:].reset_index(drop=True)

    # Convert the first row to column names
    colnames = list(df.iloc[0])

    colnames[2] = "Week Sales"
    colnames[3] = "Lag Week Sales"

    # If "Dist." in column names, replace with "Distributor"
    colnames = [col.replace("Dist.", "Distributor") for col in colnames]
    colnames = ["Cumulative" if "Cume" in col else col for col in colnames]
    colnames = [col.replace("Cume", "Cumulative") for col in colnames]

    for important_col in ["#", "Title", "Cumulative", "Weeks"]:
        if filename != "070797.htm":
            assert important_col in colnames, f"{important_col} not in {colnames}"

    df.columns = colnames
    df = df[1:]

    # Drop columns to right of "Distributor"
    if "Distributor" in df.columns:
        df = df.iloc[:, :df.columns.get_loc("Distributor") + 1]

    # Drop if missing title
    df = df.dropna(subset=["Title"])

        
    # Drop if title contains "Top 5" or "Top 10"
    df = df[~df["Title"].str.contains("Top 5|Top 10|Top 20")]

    # Drop first column
    df = df.drop(df.columns[0], axis=1)

    # Convert sales columns to numeric
    for column in ["Week Sales", "Lag Week Sales", "Cumulative", "AVG"]:
        if "AVG" not in df.columns and column == "AVG":
            continue
        df[column] = df[column].str.replace("$", "").str.replace(",", "").astype(pd.Int64Dtype())

    # Add a column for the date
    df["Date"] = filename.split(".")[0]

    # Convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%m%d%y")

    return df

def clean_title(cleaned_name):

    # Flag rereleases to change search year constraints
    rerelease = False
    if "RE" in cleaned_name or "rerelease" in cleaned_name.lower() or "reissue" in cleaned_name.lower() or "re-release" in cleaned_name.lower():
        cleaned_name = cleaned_name.replace("RE", "")
        cleaned_name = cleaned_name.replace("rerelease", "")
        cleaned_name = cleaned_name.replace("reissue", "")
        cleaned_name = cleaned_name.replace("re-release", "")
        rerelease = True

    # Remove anything in parentheses
    cleaned_name = re.sub(PAREN_REGEX, '', cleaned_name)

    # Remove anything after colon
    cleaned_name = cleaned_name.split(":")[0]

    # Remove accents
    cleaned_name = unidecode(cleaned_name)

    # Remove punctuation
    cleaned_name = cleaned_name.translate(str.maketrans('', '', string.punctuation))

    # Need to wait until after above to successfully flag RE
    cleaned_name = cleaned_name.lower()
    if "imax" in cleaned_name:
        cleaned_name = cleaned_name.replace(" imax", " ")
    cleaned_name = cleaned_name.replace(" pt", " Part")

    return cleaned_name, rerelease

# Read the data
filenames = os.listdir("html")
dfs = []
for filename in progressbar.progressbar(filenames):
    try:
        dfs.append(parse_html(filename))
    except ValueError:
        print(f"Failed to parse {filename}, No Tables")
    except:
        print(f"Failed to parse {filename}")
        raise

df = pd.concat(dfs)

df['Theaters'] = df['Theaters'].astype(pd.Int64Dtype())
df['Weeks'] = df['Weeks'].astype(pd.Int64Dtype())

cleaning_content = [clean_title(title) for title in df["Title"]]
results = [content for content in cleaning_content]

df['Clean Title'] = [content[0] for content in cleaning_content]
df['Rerelease'] = [content[1] for content in cleaning_content]

# Hash clean title into an ID
df['movie_id'] = df['Clean Title'].apply(hash)

# Disambiguate movies with the same name while computing movie spell lengths
max_spell_length = 10000
counter = 0
while max_spell_length > 60:
    print("Iteration: ", counter)
    wide_release_dates = df[df['Theaters'] >= 600].groupby('movie_id')['Date'].min()
    # Rename the column
    wide_release_dates = wide_release_dates.reset_index()
    wide_release_dates.columns = ['movie_id', 'Date_wide']

    # Drop any showings before the wide release
    df = df.merge(
        wide_release_dates, 
        on='movie_id',
    )
    
    df = df[df['Date'] >= df['Date_wide']]

    # Compute weeks since wide release
    df['weeks_since_release'] = (df['Date'] - df['Date_wide']).dt.days // 7

    # If a movie has been out for more than 50 weeks, it's likely a different movie with the same name
    # Thus, flag title with counter
    df['movie_id'] = np.where(df['weeks_since_release'] > 60, df['movie_id'] + 1, df['movie_id'])

    max_spell_length = df['weeks_since_release'].max()
    counter += 1
    
    if max_spell_length > 60:
        df = df.drop('Date_wide', axis=1)

# Rename Date_wide to Wide Release Date
df = df.rename(columns={"Date_wide": "Wide Release Date"})

# With the data cleaned, we can parse it into something normalized
weekly_sales = df[["movie_id", "Date", "Week Sales", "Theaters", "Weeks"]]
weekly_sales = weekly_sales.set_index(["movie_id", "Date"])
assert weekly_sales.index.is_unique
weekly_sales.to_parquet("data/weekly_sales.parquet")

# Similarly we can make our movies table
movies = df[["movie_id", "Clean Title", "Distributor", "Rerelease", "Wide Release Date"]]
movies = movies.drop_duplicates()
movies = movies.set_index("movie_id")
assert movies.index.is_unique
print(movies.head(17))
movies.to_parquet("data/movies.parquet")