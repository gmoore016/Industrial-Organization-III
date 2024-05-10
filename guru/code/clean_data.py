import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import progressbar
import string
from unidecode import unidecode

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

# With the data cleaned, we can parse it into something normalized
weekly_sales = df[["Title", "Date", "Week Sales", "Theaters", "Weeks"]]
weekly_sales.set_index(["Title", "Date"], inplace=True)
weekly_sales.to_parquet("data/weekly_sales.parquet")


# Movie-Level Data Cleaning
movies = df[["Title", "Distributor"]]
movies = movies.drop_duplicates()

# Let's go ahead and clean the titles now too so the cleaning is standardized
cleaning_content = [clean_title(title) for title in movies["Title"]]
movies["Clean Title"] = [content[0] for content in cleaning_content]
movies["Rerelease"] = [content[1] for content in cleaning_content]

movies.set_index("Title", inplace=True)
movies.to_parquet("data/movies.parquet")