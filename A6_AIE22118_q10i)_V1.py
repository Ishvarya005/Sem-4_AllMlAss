from sklearn.neural_network import MLPClassifier
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

def train_and_plot(X_train, y_train, activation, title):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(), activation=activation, solver='sgd', learning_rate_init=0.05, max_iter=1000)
    mlp_classifier.fit(X_train, y_train)

    # Plot loss curve
    plt.plot(mlp_classifier.loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

def main():
    # Define input and output data
    X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    y_train = np.array([0, 1, 0, 0])  # XOR gate
    #y_train= np.array([0,1,0,0]) #and gate 

    # For Question 1: Using MLPClassifier with unit-step Activation
    train_and_plot(X_train, y_train, 'identity', 'Iterations vs Loss (MLPClassifier with Identity Activation)')

    # For Question 2: Using MLPClassifier with Logistic Activation
    train_and_plot(X_train, y_train, 'logistic', 'Iterations vs Loss (MLPClassifier with Logistic Activation)')

if __name__ == "__main__":
    main()
