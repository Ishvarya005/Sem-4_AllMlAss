import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:

    def __init__(self, activation_func, learning_rate=0.05, n_iters=1000, convergence_threshold=0.002):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = activation_func
        self.weights = None
        self.bias = None
        self.convergence_threshold = convergence_threshold
        self.error_values = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for epoch in range(self.n_iters):
            total_error = 0

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

                total_error += (y_[idx] - y_predicted) ** 2

            self.error_values.append(total_error)

            if total_error <= self.convergence_threshold:
                print(f"Converged after {epoch + 1} epochs.")
                break

        if total_error > self.convergence_threshold:
            print("Did not converge after {} epochs.".format(self.n_iters))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def get_weights(self):
        return self.weights, self.bias

def main():
    # Load the dataset from the Excel file
    data = pd.read_excel("D:/Sem-4/ML/Assignments/Ass6/purchase_dataNew.xlsx", engine="openpyxl")

    # Display the first few rows of the dataset to understand its structure
    print(data.head())

    # Define the features (X) and target variable (y)
    X_train = data.drop(["High Value", "Customer"], axis=1).values
    y_train = data["High Value"].values

    # Create an instance of the Perceptron class with the unit step activation function
    perceptron_unit_step = Perceptron(activation_func=unit_step_func)

    # Train the perceptron on the dataset
    perceptron_unit_step.fit(X_train, y_train)

    # Get the weights obtained through perceptron learning
    perceptron_weights, perceptron_bias = perceptron_unit_step.get_weights()

    # Calculate weights using the pseudo-inverse method
    pseudo_inverse_weights = np.linalg.pinv(X_train) @ y_train

    # Compare the weights obtained
    print("Weights obtained through perceptron learning:", perceptron_weights, perceptron_bias)
    print("Weights obtained through pseudo-inverse method:", pseudo_inverse_weights)

if __name__ == "__main__":
    main()
