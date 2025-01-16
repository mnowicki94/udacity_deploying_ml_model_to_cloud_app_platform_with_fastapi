import pandas as pd

# Load the data
data = pd.read_csv("data/census.csv")

# Remove spaces from column names
data.columns = data.columns.str.replace(" ", "")

# Remove spaces from string values in the dataframe
for col in data.select_dtypes(["object"]).columns:
    data[col] = data[col].str.replace(" ", "")

# Save the cleaned data
data.to_csv("data/cleaned_census.csv", index=False)
