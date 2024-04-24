import pandas as pd

# Read the JSON file
data = pd.read_json('datasets/may26-www.json')

# Convert the data to CSV format
data.to_csv('datasets/may26-www.csv', index=False)