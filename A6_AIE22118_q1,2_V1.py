import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

def bipolar_step_func(x):
    return np.where(x > 0, 1, -1)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

def relu_func(x):
    return np.maximum(0, x)
##. The np.maximum() function compares two arrays element-wise and returns the element-wise 
# maximum of the two,returns the maximum of zero and the input value x.
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

        # init parameters
        self.weights = np.array([0.2, -0.75])
        self.bias = 10

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

def main():
    # For Question 1: Using Only Unit Step Activation
    perceptron_unit_step = Perceptron(activation_func=unit_step_func)
    X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    y_train = np.array([0, 1, 0, 0])  # AND gate
    perceptron_unit_step.fit(X_train, y_train)

    # Plot epochs against error values
    plt.plot(range(1, len(perceptron_unit_step.error_values) + 1), perceptron_unit_step.error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Sum-Square Error')
    plt.title('Epochs vs Error Values (Unit Step Activation)')
    plt.show()

    # For Question 2: Using rest of all Activation Functions
    perceptron_all_activations = Perceptron(activation_func=sigmoid_func)
    X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    y_train = np.array([0, 1, 0, 0])  # AND gate
    perceptron_all_activations.fit(X_train, y_train)

    # Plot epochs against error values
    plt.plot(range(1, len(perceptron_all_activations.error_values) + 1), perceptron_all_activations.error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Sum-Square Error')
    plt.title('Epochs vs Error Values')
    plt.show()

if __name__ == "__main__":
    main()
