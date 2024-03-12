from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class MLPPerceptron:
    def __init__(self, learning_rate, n_iters, convergence_threshold):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.model = None
        self.convergence_threshold = convergence_threshold
        self.error_values = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        tot_epochs = 0

        self.model = MLPClassifier(hidden_layer_sizes=(1,), activation='logistic', learning_rate_init=self.lr,
                                   max_iter=self.n_iters, tol=self.convergence_threshold, random_state=42)

        self.model.fit(X, y)

        tot_epochs = self.model.n_iter_

        return tot_epochs

    def predict(self, X):
        return self.model.predict(X)

def main():
    # Learning rates to be tested
    lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    tot_epochs = []

    for learning_rate in lr:
        # Create an instance of the MLPPerceptron class
        mlp_perceptron_model = MLPPerceptron(learning_rate, 1000, 1e-3)

        # Define the training data
        X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
        y_train = np.array([0, 1, 0, 0])

        # Train the MLPPerceptron
        epoch = mlp_perceptron_model.fit(X_train, y_train)
        tot_epochs.append(epoch)

    # Plot epochs against learning rates
    plt.plot(lr, tot_epochs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs')
    plt.title('Learning Rate Vs Epochs (MLP Perceptron)')
    plt.show()

if __name__ == "__main__":
    main()
