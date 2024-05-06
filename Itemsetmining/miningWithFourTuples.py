import pandas as pd
from collections import Counter
import os

df = pd.read_csv('merged_Friday.csv')

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

essential_columns = [' Source IP', ' Destination IP',
                     ' Destination Port', ' Protocol']  # ' Subflow Fwd Bytes',, ' Label'
# 'Total Length of Fwd Packets', ' Bwd Packet Length Min',
# ' Fwd Packet Length Mean',' Source Port']

# Drop non-essential columns
filtered_df = df[essential_columns]
print("\nFiltered dataset Summery")
print(filtered_df.info())


# Function to convert IP addresses to first subnet
def convert_to_first_subnet(ip_address):
    octets = ip_address.split('.')
    octets[2] = '0'
    octets[3] = '0'
    return '.'.join(octets)


# Strip leading and trailing spaces from 'Source IP' and 'Destination IP' columns
filtered_df[' Source IP'] = filtered_df[' Source IP'].astype(str).str.strip()
filtered_df[' Destination IP'] = filtered_df[' Destination IP'].astype(str).str.strip()

# Apply the function to 'Source IP' and 'Destination IP' columns
filtered_df[' Source IP'] = filtered_df[' Source IP'].apply(convert_to_first_subnet)
filtered_df[' Destination IP'] = filtered_df[' Destination IP'].apply(convert_to_first_subnet)

# Check the first few rows of the DataFrame to verify the changes
print("\nFiltered dataset after clearing subnets for first 2 octets")
print(filtered_df.head())
print(filtered_df.info())

with_ID = filtered_df.copy()
with_ID.insert(0, 'alertID', range(len(with_ID)))
print("\nAdding a ID number for the records")
print(with_ID.head())

dataset_with_ID = with_ID.values.tolist()

print("\nconverting the dataset in to a list")

dataset = [alert[1:] for alert in dataset_with_ID]

# Convert each sublist to a tuple for hashability
data_tuples = [tuple(sublist) for sublist in dataset]

# Count occurrences of each unique record
record_counts = Counter(data_tuples)

print("\nnumber occurrences of each unique record")
print(f"{len(record_counts) }")

print("\n\nFPMax\n")

# Assuming 'filtered_df' is your DataFrame with categorical values

# Create a copy of the DataFrame
new_df = filtered_df.copy()

# Initialize a global mapping to keep track of numerical values
global_mapping = Counter()

# Initialize dictionaries to store both forward and reverse mappings
forward_mapping = {}
reverse_mapping = {}

# Replace categorical values with numerical values using global mapping
for column in new_df.columns:
    # Get unique values in the current column
    unique_values = new_df[column].unique()

    # Map each unique value to a numerical value based on global mapping
    forward_mapping[column] = {value: global_mapping.setdefault(value, len(global_mapping)) for value in unique_values}

    # Create a reverse mapping dictionary
    reverse_mapping[column] = {num: val for val, num in forward_mapping[column].items()}

    # Replace categorical values in the column with numerical values
    new_df[column] = new_df[column].map(forward_mapping[column])

print("\nDisplay the DataFrame with numerical values")
print(new_df.head())

# # Display forward and reverse mappings for each column
# for column in new_df.columns:
#     print(f"For {column}:")
#     print("Forward Mapping:", forward_mapping[column])
#     print("Reverse Mapping:", reverse_mapping[column])
#     print()

combined_dict = {k: v for inner_dict in reverse_mapping.values() for k, v in inner_dict.items()}

item_dataset = [tuple(x) for x in new_df.to_records(index=False)]

# Add ID field to the start of each tuple
item_dataset_withID = [(i,) + record for i, record in enumerate(item_dataset)]

new_df.to_csv('data.txt', index=False, sep=' ', header=False)

# Run the algorithm

os.system("java -jar spmf.jar run FPMax data.txt output.txt 0.1%")

itemset_records_object = []
itemset_records_numbers = []

# Read the output file line by line
outFile = open("output.txt", 'r', encoding='utf-8')
for string in outFile:
    itemset = []
    parts = string.split('#SUP:')
    numbers = list(map(int, parts[0].split()))
    support_count = int(parts[1].strip())

    itemset_using_numbers = [numbers, support_count]
    itemset_records_numbers.append(itemset_using_numbers)

    # Translate numerical values to attribute names using reverse mapping
    attribute_names = [str(combined_dict[num]) for num in numbers]
    itemset = [attribute_names, support_count]
    # Output the result
    # print(f"Pattern: {' '.join(attribute_names)}, Support Count: {str(support_count)}")
    itemset_records_object.append(itemset)

outFile.close()

# Calculate the count of unique values in the 'type' field
type_counts = df[' Label'].value_counts()

print("\nDisplay the count of unique values in Label")
print(type_counts)

def return_unique_labels(alertID_List):
    # Filter DataFrame based on selected IDs
    selected_records = df.iloc[alertID_List]
    # Count unique values in a certain field (e.g., Field1) in the selected records
    unique_value_counts = selected_records[' Label'].value_counts()
    return(unique_value_counts)

for index,record in enumerate(itemset_records_numbers):
    itemset = record[0]
    containing_alerts = []
    for alert in item_dataset_withID:
        alert_items = set(alert[1:])  # Exclude the ID field
        if set(itemset).issubset(alert_items):
            containing_alerts.append(alert[0])  # Append the ID

    print(f"Pattern {index}: {itemset_records_object[index][0]}, \n{return_unique_labels(containing_alerts)}\n===============================================================================================================")

