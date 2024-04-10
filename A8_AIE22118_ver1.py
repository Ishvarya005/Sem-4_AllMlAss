import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature    # Feature used for splitting
        self.value = value        # Threshold value for splitting (for continuous features)
        self.left = left          # Left child (for binary split)
        self.right = right        # Right child (for binary split)
        self.label = label        # Predicted label (for leaf nodes)

def entropy(target_column):
    # Calculate the entropy of the target column
    unique_classes, class_counts = np.unique(target_column, return_counts=True)
    probabilities = class_counts / len(target_column)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def information_gain(data, feature, target_column):
    # Calculate the information gain for a given feature
    unique_values = data[feature].unique()
    total_entropy = entropy(target_column)
    weighted_entropy = 0
    
    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset[target_column])
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    return total_entropy - weighted_entropy

def bin_continuous_feature(feature_data, num_bins=None, binning_type='equal_width'):
    if num_bins is None:
        # Default number of bins
        num_bins = 10
        
    if binning_type == 'equal_width':
        # Equal width binning
        bin_edges = pd.cut(feature_data, bins=num_bins, retbins=True)[1]
    elif binning_type == 'frequency':
        # Frequency binning
        bin_edges = pd.qcut(feature_data, q=num_bins, retbins=True, duplicates='drop')[1]
    else:
        raise ValueError("Invalid binning type. Please choose 'equal_width' or 'frequency'.")

    # Bin the feature data based on the bin edges
    binned_feature = pd.cut(feature_data, bins=bin_edges, labels=False)
    
    return binned_feature, bin_edges

def find_root_node(data, target_column, binning_type='equal_width', num_bins=None):
    # Convert continuous-valued features to categorical
    binning_results = {}
    for column in data.columns:
        if column != target_column and data[column].dtype == 'float64':
            binned_feature, bin_edges = bin_continuous_feature(data[column], num_bins, binning_type)
            data[column] = binned_feature
            binning_results[column] = bin_edges
    
    # Find the feature with the highest information gain
    information_gains = {}
    for column in data.columns:
        if column != target_column:
            information_gains[column] = information_gain(data, column, target_column)
    
    root_node = max(information_gains, key=information_gains.get)
    print("root node : ",root_node)
    print("binning results: ",binning_results)
    return root_node, binning_results

def build_decision_tree(data, target_column, max_depth=None, binning_type='equal_width', num_bins=None):
    def _build_tree(sub_data, depth):
        if depth == 0 or len(sub_data[target_column].unique()) == 1:
            label = sub_data[target_column].mode().iloc[0]
            return Node(label=label)
        
        if len(sub_data.columns) == 1:
            label = sub_data[target_column].mode().iloc[0]
            return Node(label=label)
        
        root_feature, binning_results = find_root_node(sub_data, target_column, binning_type, num_bins)
        node = Node(feature=root_feature)
        node.binning_results = binning_results  # Include binning results
        
        left_data = sub_data[sub_data[root_feature] == 0]
        right_data = sub_data[sub_data[root_feature] == 1]
        
        if len(left_data) == 0 or len(right_data) == 0:
            label = sub_data[target_column].mode().iloc[0]
            return Node(label=label)
        
        node.left = _build_tree(left_data, depth - 1)
        node.right = _build_tree(right_data, depth - 1)
    
        return node
    
    if not pd.api.types.is_categorical_dtype(data[target_column]):
        data[target_column] = data[target_column].astype('category')
    
    if max_depth is None:
        max_depth = float('inf')
    
    decision_tree = _build_tree(data, max_depth)
    return decision_tree


def print_tree(node, indent=0):
    if node is None:
        return
    
    if node.label is not None:
        print(" " * indent + "Predicted Label:", node.label)
        return
    
    print(" " * indent + "Feature:", node.feature)
    if hasattr(node, 'binning_results'):
        print(" " * indent + "Binning Results:", node.binning_results[node.feature])
    print_tree(node.left, indent + 4)
    print_tree(node.right, indent + 4)


def main():
    file_path = input("Enter the file path of your dataset: ")
    target_column = input("Enter the name of the target column: ")

    # Load your dataset
    data = pd.read_csv(file_path)

    # Build the Decision Tree with default parameters
    dt_default = build_decision_tree(data, target_column)
    print("Decision Tree (Default Parameters):")
    print_tree(dt_default)

    # Build the Decision Tree with custom parameters
    dt_custom = build_decision_tree(data, target_column, max_depth=5, binning_type='frequency', num_bins=5)
    print("Decision Tree (Custom Parameters):")
    print_tree(dt_custom)

if __name__ == "__main__":
    main()
