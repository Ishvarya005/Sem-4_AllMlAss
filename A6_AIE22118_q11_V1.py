import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

def load_data(file_path):
    """Load preprocessed dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Split data into features and target variable."""
    X = data.drop(columns=["koi_disposition"])
    y = data["koi_disposition"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """Initialize and train MLPClassifier."""
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    mlp_classifier.fit(X_train, y_train)
    return mlp_classifier

def evaluate_model(model, X_test, y_test):
    """Predict and evaluate the model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    # Load data
    data = load_data("preprocessed_data.csv")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
