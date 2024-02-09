import pandas as pd
import math

class KNNclassifier:
    def __init__(self, k):
        # Initializing the number of neighbors (k)
        self.k = k
    
    def fit(self, X_train, y_train):
        # Storing the training data (X_train and y_train) in the class attributes,
        # making it available for later use during predictions.
        # Since training of the model doesn't really happen in K-NNs (lazy learners),
        # including fit() just to be consistent across ML algos
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, X_train):
        def findEuclidean(vec1, vec2):
            euclidean = 0
            # Finding L2 norm of the given vectors
            for i in range(len(vec2)):
                euclidean += (vec2[i] - vec1[i]) ** 2  # Finding sum of squares of the difference of the vector elements
            return math.sqrt(euclidean)  # or **0.5

        # Convert training and test data to NumPy arrays
        train_features = X_train.values
        test_features = X_test.values
        
        # List to store predictions for each test point
        predictions_list = []

        # Iterate over each test point
        for test_point in test_features:
            # List to store distances for the current test point
            distances_for_test_point = []

            # Iterate over each training point
            for train_point in train_features:
                # Calculate Euclidean distance between test and training points
                distance = findEuclidean(test_point, train_point)
                distances_for_test_point.append(distance)

            # 1. Identify the indices of k-nearest neighbors based on distances and sort them in ascending order
            sorted_indices = sorted(range(len(distances_for_test_point)), key=lambda k: distances_for_test_point[k])
            # Extract first k indices
            k_nearest_indices = sorted_indices[:self.k]

            # 2. Retrieve the labels of k-nearest neighbors from the training set
            k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]

            # 3. Use majority voting to determine the predicted label
            unique_labels = set(k_nearest_labels)
            prediction = max(unique_labels, key=k_nearest_labels.count)

            # 4. Append the prediction to the list of predictions for each test point
            predictions_list.append(prediction)

        # 5. Return the list of predictions for each test point
        return predictions_list

# Load training and testing datasets
train_data = pd.read_csv("train1_Mobile.csv")
test_data = pd.read_csv("test1_Mobile.csv")

X_train = train_data.drop("price_range", axis=1)  # Features
y_train = train_data["price_range"]  # Labels

X_test = test_data.drop('price_range', axis=1)
y_test = test_data['price_range']

# Creating the KNN classifier object
knnClassifier = KNNclassifier(k=7)

# Fitting the model on the training data
knnClassifier.fit(X_train, y_train)

# Make predictions on the test set
test_predictions = knnClassifier.predict(X_test, X_train)

# Print predictions
print("Predictions on the test set are as follows:")
print(test_predictions)

# Calculate accuracy
correct_pred = 0
for i in range(len(test_predictions)):
    if test_predictions[i] == y_test.iloc[i]:
        correct_pred += 1

# Calculate accuracy as the ratio of correct predictions to the total number of predictions
accuracy = correct_pred / len(test_predictions) * 100

print(f"Accuracy: {accuracy:.2f}%")
