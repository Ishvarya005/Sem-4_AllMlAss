import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from sklearn.model_selection import RandomizedSearchCV

def load_data(file_path):
    """Load preprocessed dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Split data into features and target variable."""
    X = data.drop(columns=["koi_disposition"])
    y = data["koi_disposition"]
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model1(X_train, y_train):
    """Initialize and train MLPClassifier."""
    rf_classifier = RandomForestClassifier(max_depth=5,n_estimators=150 ,random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model1(model, X_test, y_test):
    """Predict and evaluate the model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
def train_model2(X_train, y_train):
    """Initialize and train MLPClassifier."""
    xgboost_classifier = GradientBoostingClassifier(learning_rate=0.01,max_depth=5,n_estimators=150 ,random_state=42)
    xgboost_classifier.fit(X_train, y_train)
    return xgboost_classifier

def evaluate_model2(model, X_test, y_test):
    """Predict and evaluate the model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


from sklearn.neural_network import MLPClassifier

def train_model3(X_train, y_train):
    """Initialize and train MLPClassifier."""
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(150, 50), activation='relu', solver='adam', random_state=42)
    param_dist = {
        'hidden_layer_sizes': [(150, 50), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
    }
    random_search = RandomizedSearchCV(mlp_classifier, param_dist, n_iter=10, cv=8, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search,random_search.best_params_

def evaluate_model3(model, X_test, y_test):
    """Predict and evaluate the model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


from sklearn.linear_model import Perceptron

def train_model4(X_train, y_train):
    """Initialize and train perceptron"""
    perceptron = Perceptron()
    param_dist = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 200, 300],
        'tol': [1e-3, 1e-4, 1e-5],
    }
    random_search = RandomizedSearchCV(perceptron, param_dist, n_iter=10, cv=8, random_state=42)
    #here n_iter is for the no. of combinations that the randomsearch would try with the parameters specified in the param_dist
    #and cv is the cross validation to be applied 
    random_search.fit(X_train, y_train)
    return random_search,random_search.best_params_

def evaluate_model4(model, X_test, y_test):
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
    
    #question -3 
    print("training using RandomForestClassifier")
    # Train the model
    model1 = train_model1(X_train, y_train)

    # Evaluate the model
    evaluate_model1(model1, X_test, y_test)
    
    print("training using GradientBoostClassifier")
    # Train the model
    model2 = train_model2(X_train, y_train)

    # Evaluate the model
    evaluate_model2(model2, X_test, y_test)
    
    #question-2 
    print("Hyperparameter tuning using randomsearchcv for mlp classifier : ")
    # Train the model
    model3,params = train_model3(X_train, y_train)
    print("Best parameters after using randomsearchcv : ",params)

    # Evaluate the model
    evaluate_model3(model3, X_test, y_test)

    print("Hyperparameter tuning using randomsearchcv for perceptron: ")
    # Train the model
    model4,params = train_model4(X_train, y_train)
    print("Best parameters after using randomsearchcv : ",params)
    # Evaluate the model
    evaluate_model4(model4, X_test, y_test)

if __name__ == "__main__":
    main()
