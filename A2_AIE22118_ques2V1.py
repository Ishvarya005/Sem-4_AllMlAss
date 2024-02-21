import pandas as pd
#library for data manipulation and analysis

class KNNClassifierEuclidean:
    def __init__(self, k=1): 
        #initialisation
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Stores the training data  in the KNNClassifierEuclidean object.
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, point1, point2):
    # Ensure that point1 and point2 have the same dimensions
        min_len = min(len(point1), len(point2))
    
    # Trim or pad the vectors to have the same length
        point1 = point1[:min_len]
        point2 = point2[:min_len]

    # Calculates the Euclidean distance between two points
        distance = ((point1 - point2) ** 2).sum() ** 0.5
        return distance


    def maxSort(self, distances):
        # 
        indices_with_distances = list(enumerate(distances))
        sorted_indices = sorted(indices_with_distances, key=lambda x: x[1])
        sorted_indices_only = [index for index, _ in sorted_indices]
        return sorted_indices_only

    def predict(self, X_test):
        # Make predictions on the test set
        predictions = []
        for test_point in X_test.values:
            # Calculate distances between test_point and all points in X_train using Euclidean distance
            distances = [self.euclidean_distance(test_point, train_point) for train_point in self.X_train.values]

            # Get indices of k-nearest neighbors without using NumPy
            k_nearest_indices = self.maxSort(distances)[:self.k]

            # Get labels of k-nearest neighbors
            k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]

            # Make a prediction based on majority vote
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)

        return predictions

# Example usage:
# Assuming you have a dataset with train_data and test_data
train_data = pd.read_csv("mobile_train.csv")
test_data = pd.read_csv("mobile_test.csv")

X_train = train_data.drop('price_range', axis=1)
y_train = train_data['price_range']

X_test = test_data

# Create KNN classifier with Euclidean distance
knn_classifier_euclidean = KNNClassifierEuclidean(k=3)

# Fit the model on the training data
knn_classifier_euclidean.fit(X_train, y_train)

# Make predictions on the test set
test_predictions_euclidean = knn_classifier_euclidean.predict(X_test)

# Print predictions
print('Predictions on the test set (Euclidean distance):')
print(test_predictions_euclidean)


from sklearn.metrics import accuracy_score

# Assuming 'price_range' is the actual labels in your test_data
actual_labels = test_data['price_range']

# Calculate accuracy
accuracy = accuracy_score(actual_labels, test_predictions_euclidean)

# Print the accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')
