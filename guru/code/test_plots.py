import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("data/weekly_sales.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Convert week sales to int
df["Week Sales"] = df["Week Sales"].str.replace("$", "").str.replace(",", "").astype(int)
df["ln Week Sales"] = np.log(df["Week Sales"])

# Histogram of weekly sales
plt.hist(df["ln Week Sales"], bins=20)
plt.title("Weekly Sales")
plt.xlabel("ln Week Sales")
plt.ylabel("Count")
plt.show()


# Sum revenue by week
weekly_revenue = df.groupby("Date")["Week Sales"].sum()
ln_weekly_revenue = np.log(weekly_revenue)

# Plot the data
plt.plot(ln_weekly_revenue)
plt.title("Revenue per Week")
plt.xlabel("Date")
plt.show()

# Open movie info
movie_info = pd.read_csv("data/movies.csv")

# Bar chart of top 10 distributors
distributor_counts = movie_info["Distributor"].value_counts()
top_10 = distributor_counts.head(10)
top_10.plot(kind="bar")
# Increase plot area
plt.tight_layout()
plt.title("Top 10 Distributors")
plt.xlabel("Distributor")
plt.ylabel("Count")
plt.show()