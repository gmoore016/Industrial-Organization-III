import requests 
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
import os
import progressbar

ARCHIVES_ROOT = "http://www.boxofficeguru.com/archives2.htm"
URL_ROOT = "http://www.boxofficeguru.com/"
DATE_REGEX = re.compile(r"\w+\.? \d{1,2} - (\w+\.? )?\d{1,2}")

# Pull the archives page
homepage = requests.get(ARCHIVES_ROOT)

# Replace line breaks with spaces
homepage = homepage.text.replace("\n", " ")

# Compress multiple spaces to one
homepage = re.sub(r"\s+", " ", homepage)

# Get the set of links with text matching the date regex
homepage_html = BeautifulSoup(homepage, "html.parser")
links = homepage_html.find_all("a", string=DATE_REGEX)

# Extract the href attribute from each link
links = [link["href"] for link in links]

# Extract numeric part of each link
links = [re.search(r"\d{6}.htm", link).group() for link in links]

# Get set of files already scraped
scraped_files = os.listdir("html")

# Filter out links that have already been scraped
links = [link for link in links if link not in scraped_files]

for link in progressbar.progressbar(links):
    # Pause a random amount between pulls
    time.sleep(random.uniform(1, 2))

    url = URL_ROOT + link

    response = requests.get(url)

    # Write the response to a file
    with open("html/" + link, "w", encoding='utf-8') as file:
        file.write(response.text)