import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
     #to scale the weights between (-1,1) so that the learning is faster 
    weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    return weights_input_hidden, weights_hidden_output

def train_neural_network(X, y, learning_rate, n_iterations, convergence_threshold):
    input_size = X.shape[1]#to get no. of cols --input features
    hidden_size = 2
    output_size = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for iteration in range(n_iterations):
        # Forward propagation
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Calculate error
        error = y - predicted_output

        # Check for convergence
        if np.mean(np.abs(error)) <= convergence_threshold:
            print(f"Converged after {iteration + 1} iterations.")
            break

        # Backpropagation
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights
        weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_error)
        weights_input_hidden += learning_rate * X.T.dot(hidden_layer_error)

    return weights_input_hidden, weights_hidden_output

def main():
    # Training data for XOR gate
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    learning_rate = 0.05
    n_iterations = 1000
    convergence_threshold = 0.002

    trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(X_train, y_train, learning_rate, n_iterations, convergence_threshold)

    # Test the trained model
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predicted_output = sigmoid(sigmoid(test_data.dot(trained_weights_input_hidden)).dot(trained_weights_hidden_output))
    print("Predicted Output:")
    print(predicted_output)

if __name__ == "__main__":
    main()
