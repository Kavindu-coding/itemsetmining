import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import json
from mining_functions import adaptive_bin_handling

df = pd.read_csv('NewDataSet.csv')

# Get the number of unique items in each column
unique_counts = df.nunique()
print("Unique value count for each feature")
# Print header names with the number of unique items
for column_name, count in unique_counts.items():
    print(f"{column_name}: {count} unique items")

print("\nDataframe Sample")
print(df.head())

print("\nDataframe Summery")
print(df.info())

bin_data = []

selected_features = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                     'Flow IAT Mean',
                     'Pkt Size Avg',
                     'Fwd Act Data Pkts',
                     'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Fwd Pkts/s',
                     'SYN Flag Cnt',
                     'ACK Flag Cnt',
                     'Subflow Bwd Byts', 'PSH Flag Cnt'
                     ]

handled_columns = []
unhandled_columns = []

for col in selected_features:
    if (len(df[col].value_counts())) > 20:
        handled_columns.append(col)
    else:
        unhandled_columns.append(col)

essential_columns = []

for to_be_handled_column_names in handled_columns:
    category_name, bin_data = adaptive_bin_handling(to_be_handled_column_names, df, bin_data)
    essential_columns.append(category_name)

other_columns = ['Dst Port','Protocol']
essential_columns = essential_columns + other_columns + unhandled_columns

# Drop non-essential columns
filtered_df = df[essential_columns]
print("\nFiltered dataset Summery")
print(filtered_df.info())

categorical_cols = filtered_df.select_dtypes(include=['object','category']).columns
numerical_cols = filtered_df.select_dtypes(include=['int64']).columns

# Encode categorical features
encoder = OrdinalEncoder()
encoder.fit_transform(filtered_df[categorical_cols])
# Save the encoder and the model

for i, category in enumerate(encoder.categories_):
    if np.nan not in category:
        encoder.categories_[i] = np.append(category, np.nan)

joblib.dump(encoder, 'feature_encoder.pkl')

with open('categorization.json', 'w') as file:
    json.dump(bin_data, file)

# Save the column order
column_order = filtered_df.columns.tolist()
with open('column_order.json', 'w') as file:
    json.dump(column_order, file)

with_ID = filtered_df.copy()
with_ID.insert(0, 'alertID', range(len(with_ID)))
print("\nAdding a ID number for the records")
print(with_ID.head())

dataset_with_ID = with_ID.values.tolist()

print("\nconverting the dataset in to a list")
dataset = [alert[1:] for alert in dataset_with_ID]

# Convert each sublist to a tuple for hash ability
data_tuples = [tuple(sublist) for sublist in dataset]

# Count occurrences of each unique record
record_counts = Counter(data_tuples)

print("\nnumber occurrences of each unique record")
print(f"{len(record_counts) }")

# Assuming 'filtered_df' is your DataFrame with categorical values

# Initialize dictionaries to store both forward and reverse mappings
forward_mapping = {}
reverse_mapping = {}

# Initialize the global counter to keep track of numerical values
global_counter = 0

# Iterate over each column in the DataFrame
for column in filtered_df.columns:
    # Initialize a local counter for each column
    local_counter = 0

    # Initialize dictionaries for forward and reverse mappings for the current column
    forward_mapping[column] = {}
    reverse_mapping[column] = {}

    # Iterate over each unique value in the current column
    for value in filtered_df[column].unique():
        # Map each unique value to a numerical value based on the global counter
        forward_mapping[column][value] = global_counter + local_counter
        reverse_mapping[column][global_counter + local_counter] = value

        # Increment the local counter
        local_counter += 1

    # Update the global counter to continue numbering from the last value of the previous dictionary
    global_counter += local_counter

# Create a new DataFrame with mapped values
new_df = pd.DataFrame()

# Iterate over each column in the original DataFrame and fill in values in the new DataFrame
for column in filtered_df.columns:
    new_df[column] = filtered_df[column].map(forward_mapping[column])

combined_dict = {k: v for inner_dict in reverse_mapping.values() for k, v in inner_dict.items()}

item_dataset = [tuple(x) for x in new_df.to_records(index=False)]

# Add ID field to the start of each tuple
item_dataset_withID = [(i,) + record for i, record in enumerate(item_dataset)]

new_df.to_csv('data.txt', index=False, sep=' ', header=False)