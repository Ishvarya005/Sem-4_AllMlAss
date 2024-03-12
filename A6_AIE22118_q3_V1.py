import numpy as np
import matplotlib.pyplot as plt

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate, n_iters, convergence_threshold):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        self.convergence_threshold = convergence_threshold
        self.error_values = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        tot_epochs = 0
        #initialising using given weights
        self.weights = np.array([0.2, -0.75])
        self.bias = 10
        y_ = np.where(y > 0, 1, 0)

        for epoch in range(self.n_iters):
            total_error = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                #perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

                total_error += (y_[idx] - y_predicted) ** 2

            self.error_values.append(total_error)

            if total_error <= self.convergence_threshold:
                tot_epochs = epoch + 1
                return tot_epochs

        if total_error > self.convergence_threshold:
            tot_epochs = 0
            return tot_epochs

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

def main():
    # Learning rates to be tested
    lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    tot_epochs = []

    for learning_rate in lr:
        # To create an instance of the Perceptron class
        perceptron_model = Perceptron(learning_rate, 1000, 0.002)

        # Defining the training data
        X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
        y_train = np.array([0, 1, 0, 0])  # AND

        # Training the perceptron
        epoch = perceptron_model.fit(X_train, y_train)
        tot_epochs.append(epoch)

    # Plotting epochs against learning rates
    plt.plot(lr, tot_epochs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs')
    plt.title('Learning Rate Vs Epochs')
    plt.show()

if __name__ == "__main__":
    main()
