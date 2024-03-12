import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path, sheet_name):
    # Load the data from a specific sheet in the Excel file
    return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)

def mark_category(df):
    # Mark customers as RICH or POOR based on payments
    df['Category'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')
    return df

def split_data(df, X_columns, y_column, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = df[X_columns].values
    y = df[y_column].values
    
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_knn_classifier(k=3):
    # Create KNN classifier
    return KNeighborsClassifier(n_neighbors=k)

def evaluate_model(classifier, X_test, y_test):
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

def main():
    file_path = "D:/Sem-4/ML/Assignments/Ass3/Lab Session1 Data.xlsx"
    sheet_name = 'Purchase data'

    # Load data
    df = load_data(file_path, sheet_name)

    # Mark category
    df = mark_category(df)

    # Define columns
    X_columns = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    y_column = 'Category'

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, X_columns, y_column)

    # Create KNN classifier
    knn_classifier = create_knn_classifier(k=3)

    # Fit the model on the training data
    knn_classifier.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(knn_classifier, X_test, y_test)

if __name__ == "__main__":
    main()
