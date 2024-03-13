import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#q2:
def evaluate_linear_regression(df):
    # Step 2: One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=['Day', 'Month'])

    # Step 3: Preprocess 'Volume' column with unit conversion
    df_encoded['Volume'] = df_encoded['Volume'].replace('[KMB]+$', '', regex=True)
    df_encoded['Volume'] = df_encoded.apply(lambda row: convert_volume(row['Volume']), axis=1)

    # Step 4: Preprocess 'Chg%' column
    df_encoded['Chg%'] = df_encoded['Chg%'].astype(str).str.rstrip('%').astype('float') / 100.0

    # Step 5: Split data into features (X) and target variable (y)
    X_cols = ['Open', 'High', 'Low', 'Volume', 'Chg%', 'Day_Tue', 'Day_Mon', 'Day_Fri', 'Day_Thu', 'Day_Wed', 'Month_Jun']
    X = df_encoded[X_cols]
    y = df_encoded['Price']

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Create a linear regression model
    model = LinearRegression()

    # Step 8: Fit the model
    model.fit(X_train, y_train)

    # Step 9: Make predictions
    y_pred = model.predict(X_test)

    # Step 10: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
    print(f'R-squared (R2): {r2}')

# Function to convert Volume values
def convert_volume(value):
    multiplier = 1.0
    if value.endswith('M'):
        multiplier = 10**6
    elif value.endswith('K'):
        multiplier = 10**3
    return float(value[:-1]) * multiplier

def generate_training_data():
    
    # Generate random training data
    train_X = np.random.uniform(1, 10, size=(20, 1))
    train_Y = np.random.uniform(1, 10, size=(20, 1))
    
    # Assign random classes (0 or 1) to training data
    train_classes = np.random.choice([0, 1], size=(20,), p=[0.5, 0.5])
    
    df= pd.DataFrame({'X':train_X.reshape(-1),'Y':train_Y.flatten(),'class':train_classes},index=None)
    return train_X, train_Y, train_classes,df

def plot_training_data(train_X, train_Y, train_classes):
    # Scatter plot of training data with different colors for each class
    plt.scatter(train_X[train_classes == 0], train_Y[train_classes == 0], color='blue', label='Class 0')
    plt.scatter(train_X[train_classes == 1], train_Y[train_classes == 1], color='red', label='Class 1')
    
    # Set labels and title
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('Scatter Plot of Training Data')
    
    # Display legend
    plt.legend()
    
    # Show the plot
    plt.show()

def generate_test_data():
    # Generate a grid of test data points
    test_X, test_Y = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    test_data = np.column_stack((test_X.flatten(), test_Y.flatten()))
    
    return test_data

def classify_and_plot_test_data(train_X, train_Y, train_classes, test_data,k):
    # Create kNN classifier with k=3
    knn_classifier = KNeighborsClassifier(k)
    
    # Train the classifier using training data
    knn_classifier.fit(np.column_stack((train_X, train_Y)), train_classes)
    
    # Predict classes for test data
    predicted_classes = knn_classifier.predict(test_data)

    # Scatter plot of test data points with predicted classes
    plt.scatter(test_data[predicted_classes == 0, 0], test_data[predicted_classes == 0, 1], color='blue', alpha=0.2, label='Predicted class 0')
    plt.scatter(test_data[predicted_classes == 1, 0], test_data[predicted_classes == 1, 1], color='red', alpha=0.2, label='Predicted class 1')

    # Scatter plot of training data for reference
    plt.scatter(train_X[train_classes == 0], train_Y[train_classes == 0], color='blue', label='Training class 0')
    plt.scatter(train_X[train_classes == 1], train_Y[train_classes == 1], color='red', label='Training class 1')

    # Set labels and title
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title(f'Scatter Plot of Test Data with kNN Classification with k:{k}')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def main():
#q2
    #step1: load data
    df = pd.read_excel("D:\\Sem-4\\ML\\Assignments\\Ass5\\PurchaseData.xlsx",  sheet_name=['IRCTC Stock Price'])
    price_df= df.get('IRCTC Stock Price')
#q3
    # Call the function for linear regression evaluation
    evaluate_linear_regression(price_df)
    #  Generate training data
    train_X, train_Y, train_classes,df1 = generate_training_data()
    print(df1)
    # Plot the training data
    plot_training_data(train_X, train_Y, train_classes)
#q4
    #  Generate test data
    test_data = generate_test_data()
#q5
    # Classify test data using kNN and plot the results
    for k in range(1,15):
        classify_and_plot_test_data(train_X, train_Y, train_classes, test_data,k)

if __name__ == "__main__":
    main()
