import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import json
from mining_functions import get_max_and_min, adaptive_bin_handling

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
    category_name = adaptive_bin_handling(to_be_handled_column_names, df, bin_data)
    essential_columns.append(category_name)
