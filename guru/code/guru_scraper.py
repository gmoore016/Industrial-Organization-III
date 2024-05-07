import requests 
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random

ARCHIVES_ROOT = "http://www.boxofficeguru.com/archives2.htm"
URL_ROOT = "http://www.boxofficeguru.com/"
DATE_REGEX = re.compile(r"\w+\. \d{1,2} - \d{1,2}")

def parse_html(filename):
    with open(f"html/{filename}") as file:
        html = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Get the table element
    table = soup.find("table")

    # Convert the table to a DataFrame
    df = pd.read_html(
        str(table),
        match="\$",
    )[0]

    converters = {c: lambda x: str(x) for c in df.columns}
    df = pd.read_html(
        str(table),
        match="\$",
        converters=converters
    )[0]

    # Convert all columns to strings
    df = df.applymap(str)

    # Drop first row
    df = df[1:]

    # Convert the first row to column names
    colnames = df.iloc[0]
    colnames[2] = "Week Sales"
    colnames[3] = "Lag Week Sales"
    df.columns = colnames
    df = df[1:]

    # Replace string nan with actual NaN
    df = df.replace("nan", pd.NA)

    # Drop if missing title
    df = df.dropna(subset=["Title"])

    # Drop if title contains "Top 5" or "Top 10"
    df = df[~df["Title"].str.contains("Top 5|Top 10")]

    # Drop first column
    df = df.drop(df.columns[0], axis=1)

    # Convert sales columns to numeric
    for column in ["Week Sales", "Lag Week Sales", "Cumulative", "AVG"]:
        df[column] = df[column].str.replace("$", "").str.replace(",", "").astype(pd.Int64Dtype())

    # Add a column for the date
    df["Date"] = filename.split(".")[0]

    # Convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%m%d%y")

    return df

# Pull the archives page
homepage = requests.get(ARCHIVES_ROOT)

# Get the set of links with text matching the date regex
homepage_html = BeautifulSoup(homepage.text, "html.parser")
links = homepage_html.find_all("a", text=DATE_REGEX)

# Extract the href attribute from each link
links = [link["href"] for link in links]

for link in links:
    # Pause a random amount between pulls
    time.sleep(random.uniform(1, 2))

    url = URL_ROOT + link

    response = requests.get(url)

    # Write the response to a file
    with open("html/" + link, "w") as file:
        file.write(response.text)

filenames = ["010824.htm", "011524.htm", "031124.htm"]
dfs = [parse_html(filename) for filename in filenames]

df = pd.concat(dfs)

print(df)

