import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Find the maximum value in a column
def get_max_and_min(column_name, df):
    maximum_value = df[column_name].max()

    # Find the minimum value in a column
    minimum_value = df[column_name].min()

    # Print the maximum and minimum values
    print("Maximum Value:", maximum_value)
    print("Minimum Value:", minimum_value)


# Function to categorize values based on ranges
def categorize_value(value, ranges):
    # if pd.isnull(value):
    # return 'unknown'  # Treat NaN values as 'unknown'
    for category, (lower, upper) in ranges.items():
        if lower <= value <= upper:
            return category
    # return 'unknown'


def adaptive_bin_handling(column_name, df, bin_data):
    bin_ranges, bin_edges = pd.qcut(df[f'{column_name}'], 10, labels=None, retbins=True, precision=2, duplicates='drop')
    # print(bin_ranges.value_counts())

    ranges = {}
    for idx in range(len(bin_edges) - 1):
        ranges.update({f"{(bin_edges[idx] + bin_edges[idx + 1]) / 2}": (bin_edges[idx], bin_edges[idx + 1])})

    bin_data.append([column_name, ranges])

    # Apply categorization to 'Flow Duration' column
    df[f'{column_name} Category'] = df[f'{column_name}'].apply(lambda x: categorize_value(x, ranges))
    # print(df[f'{column_name} Category'].value_counts())

    # plt.figure(figsize=(20, 8))
    # sns.histplot(data=df, x=df[f'{column_name} Category'], hue='Label', multiple='stack', discrete='True')
    # plt.show()

    # print("\n================================================================================\n")

    return f"{column_name} Category", bin_data
