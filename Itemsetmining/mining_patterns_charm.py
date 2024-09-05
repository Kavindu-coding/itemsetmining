import pandas as pd
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
# print("Unique value count for each feature")
# # Print header names with the number of unique items
# for column_name, count in unique_counts.items():
#     print(f"{column_name}: {count} unique items")
#
# print("\nDataframe Sample")
# print(df.head())
#
# print("\nDataframe Summery")
# print(df.info())

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

other_columns = ['Dst Port', 'Protocol']
essential_columns = essential_columns + other_columns + unhandled_columns

# Drop non-essential columns
filtered_df = df[essential_columns]

# print("\nFiltered dataset Summery")
# print(filtered_df.info())

categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns
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
# print("\nAdding a ID number for the records")
# print(with_ID.head())

dataset_with_ID = with_ID.values.tolist()

print("\nconverting the dataset in to a list")
dataset = [alert[1:] for alert in dataset_with_ID]

# Convert each sublist to a tuple for hash ability
data_tuples = [tuple(sublist) for sublist in dataset]

# Count occurrences of each unique record
record_counts = Counter(data_tuples)

# print("\nnumber occurrences of each unique record")
# print(f"{len(record_counts) }")

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

# Run the algorithm
os.system("java -jar spmf.jar run Charm_bitset data.txt output.txt 0.01% true")

# Open the file in read mode
with open("output.txt", 'r', encoding='utf-8') as outFile:
    # Initialize the counter directly from enumerate
    line_count = sum(1 for _ in outFile)

# Print the number of lines
print(f"Number of patterns processing : {line_count}")

itemset_records_object = []
itemset_records_numbers = []
itemset_records_ID_list = []

# Read the output file line by line
outFile = open("output.txt", 'r', encoding='utf-8')

for string in outFile:
    itemset = []
    parts = string.split('#SUP:')
    numbers = list(map(int, parts[0].split()))
    ID_Sup_count = parts[1].split('#TID:')
    support_count = int(ID_Sup_count[0].strip())
    IDs = list(map(int, ID_Sup_count[1].split()))

    if len(numbers) <= 15:
        continue

    itemset_using_numbers = [numbers, support_count]
    itemset_records_ID_list.append(IDs)
    itemset_records_numbers.append(itemset_using_numbers)

    # Translate numerical values to attribute names using reverse mapping
    attribute_names = [str(combined_dict[num]) for num in numbers]
    itemset = [attribute_names, support_count]
    # Output the result
    # print(f"Pattern: {' '.join(attribute_names)}, Support Count: {str(support_count)}")
    itemset_records_object.append(itemset)

outFile.close()

print(f"There are {len(itemset_records_numbers)} quality patterns")


def return_unique_labels(alertID_List):
    # Filter DataFrame based on selected IDs
    selected_records = df.iloc[alertID_List]
    # Count unique values in a certain field (e.g., Field1) in the selected records
    unique_value_counts = selected_records['Label'].value_counts().to_dict()
    return unique_value_counts


alerts_with_patterns_df = pd.DataFrame()
pattern_label_list = []
alerts_with_patterns_IDs = set()

for index, IDs_record in enumerate(itemset_records_ID_list):
    alerts_with_patterns_IDs.update(IDs_record)
    pattern_label_record = list(return_unique_labels(IDs_record).items())
    pattern_label_list.append((itemset_records_object[index][0], pattern_label_record))
    # print(f"Pattern {index}: {itemset_records_object[index][0]}, \n{pattern_label_record}\n===============================================================================================================")


# Define function to get field and value
def get_field_and_value(num):
    for field, mapping in reverse_mapping.items():
        if num in mapping:
            return field, mapping[num]
    return "Unknown", "Unknown"


# Initialize a list to store the patterns with original field and value
pattern_record = pd.DataFrame(columns=['Support Count', 'Label'])
patterns_with_fields = []

suitable_list = []
not_suitable = []

for index, record in enumerate(itemset_records_numbers):
    # Split the line into items and support count
    items = record[0]
    support_count = int(record[1])

    # # Skip patterns with 3 or fewer features
    # if len(items) <= 7:
    #     continue

    # Initialize a list to store the original field and value of each item
    record_with_field = []

    # Map each item in the pattern to its original field and value
    for item in items:
        # Get the field and value using the reverse mapping dictionaries
        field, value = get_field_and_value(item)
        record_with_field.append({"field": field, "value": value})

        # Check if the field already exists in the DataFrame
        if field not in pattern_record.columns:
            # If not, add a new column with the field name and fill with NaN
            pattern_record[field] = pd.NA

        # Add the value to the corresponding field
        pattern_record.at[index, field] = value

    # Add the support_count to the pattern_data dictionary
    pattern_record.at[index, 'Support Count'] = support_count

    if len(pattern_label_list[index][1]) == 1:
        pattern_record.at[index, 'Label'] = pattern_label_list[index][1][0][0]
        # Store the pattern with original field and value along with the support count
        patterns_with_fields.append({"pattern": record_with_field, "support_count": support_count})
    else:
        label_percentage_list = []
        for multiple_labels in pattern_label_list[index][1]:
            label_percentage = (multiple_labels[1] / support_count) * 100
            label_percentage_list.append(label_percentage)

        if max(label_percentage_list) >= 75.0:
            suitable_list.append(index)
            pattern_record.at[index, 'Label'] = \
            pattern_label_list[index][1][label_percentage_list.index(max(label_percentage_list))][0]
            # Store the pattern with original field and value along with the support count
            temp_support_count = pattern_label_list[index][1][label_percentage_list.index(max(label_percentage_list))][1]
            patterns_with_fields.append({"pattern": record_with_field, "support_count": temp_support_count})
        else:
            not_suitable.append(index)
            pattern_record.at[index, 'Label'] = 'Mixed Labels-Not suitable'
            pattern_label_list[index][1][label_percentage_list.index(max(label_percentage_list))][0]

    # Store the pattern with original field and value along with the support count
    patterns_with_fields.append({"pattern": record_with_field, "support_count": support_count})

pattern_record = pattern_record.fillna('NaN').infer_objects(copy=False)

# Print the patterns with original field and value
# for pattern_info in patterns_with_fields:
#     print(pattern_info)
#     print()

pattern_record1 = pattern_record.drop(pattern_record[pattern_record['Label'] == 'Mixed Labels-Not suitable'].index)
# Save the pattern record DataFrame to a CSV file
pattern_record1.to_csv('IDS_data_0.01_3Null_19features.csv', index=False)

print(pattern_record1.info())

print(f"number of patterns {len(pattern_record)}")
print(pattern_record['Label'].value_counts())
