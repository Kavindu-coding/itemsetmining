from sklearn.metrics import accuracy_score, v_measure_score, silhouette_score, davies_bouldin_score
import numpy as np


def plot_graph_evaluate(clusters, null_handled_patterns, null_handled_patterns_encoded, df):
    print('\n')
    null_handled_patterns['Cluster'] = clusters
    null_handled_patterns['Label'] = df['Label']

    cluster_labels = null_handled_patterns['Cluster'].unique()
    label_mapping = {}

    for cluster in cluster_labels:
        unique_value_count = null_handled_patterns[null_handled_patterns['Cluster'] == cluster][
            'Label'].value_counts().reset_index().rename(columns={"index": "value", 0: "count"})
        new_cluster_name = unique_value_count.iloc[unique_value_count['count'].idxmax()]['Label']
        # print(new_cluster_name)
        label_mapping.update({cluster: new_cluster_name})

    null_handled_patterns['new_label'] = null_handled_patterns['Cluster'].map(label_mapping)

    true_labels = null_handled_patterns['Label']
    predict_label = null_handled_patterns['new_label']

    # Compute the Accuracy score
    score_1 = accuracy_score(true_labels, predict_label)
    score_2 = v_measure_score(true_labels, predict_label)
    score_3 = silhouette_score(null_handled_patterns_encoded, clusters)
    score_4 = davies_bouldin_score(null_handled_patterns_encoded, clusters)

    print(f"Accuracy Score: {score_1}")
    print(f"V Measure Score: {score_2}")
    print(f"Silhouette Score: {score_3}")
    print(f"Davies-Bouldin Index: {score_4}")

    # clusters = null_handled_patterns['Cluster'].unique()
    # num_clusters = len(clusters)
    # num_rows = (num_clusters + 3) // 4  # Calculate number of rows needed, 4 plots per row

    # fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    # fig.tight_layout(pad=5.0)

    # # Flatten axes for easier iteration
    # axes = axes.flatten()

    # for i, cluster in enumerate(clusters):
    #     cluster_data = null_handled_patterns[null_handled_patterns['Cluster'] == cluster]
    #     sns.countplot(data=cluster_data, x='Label', ax=axes[i])
    #     axes[i].set_title(f'Cluster {cluster} - Label Count')
    #     axes[i].set_xlabel('Label')
    #     axes[i].set_ylabel('Count')

    # # Hide any unused subplots
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # plt.show()  # Show the plot
    print("--------------------------------------------------------------------------------")

    return score_1, score_2, score_3, score_4


def generate_floats_between(float_num1, float_num2):
    # Generate 20 evenly spaced float values between the two given float numbers
    float_values = np.linspace(float_num1, float_num2, num=100)

    # Round the float values to 3 decimal places
    float_values_rounded = np.round(float_values, decimals=3)

    # Remove values that are equivalent to integers (those that end with .000)
    float_values_filtered = [val for val in float_values_rounded if val % 1 != 0]

    # Return the filtered list of unique float values
    return list(np.unique(float_values_filtered))
