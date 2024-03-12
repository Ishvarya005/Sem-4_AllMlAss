import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
def load_data(file_path, skip_rows):
    return pd.read_csv(file_path, skiprows=skip_rows)

def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)
#inplace=true : When you set inplace=True in a function call, you're instructing the function to apply changes directly to the object it operates on, rather than creating
# a new object with the modifications and returning it.


def one_hot_encoding(df, class_label_column):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove(class_label_column)
    return pd.get_dummies(df, columns=categorical_columns)

def handle_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            fill_value = df[column].mean()
            df[column].fillna(fill_value, inplace=True)
    return df

def save_preprocessed_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")

def calculate_mean_and_spread(df, class_label_column):
    class_labels = df[class_label_column].unique()
    class_centroids = {}
    class_spreads = {}

    for label in class_labels:
        class_data = df[df[class_label_column] == label]
        class_data_numeric = class_data.apply(pd.to_numeric, errors='coerce')

        class_centroid = class_data_numeric.mean(axis=0)
        class_spread = class_data_numeric.std(axis=0)

        class_centroids[label] = class_centroid
        class_spreads[label] = class_spread

        print(f"\nClass: {label}")
        print("Mean (Centroid):")
        print(class_centroid)
        print("Spread (Standard Deviation):")
        print(class_spread)

def plot_histogram_mean_variance(df, feature_of_interest):
    feature_data = df[feature_of_interest].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(feature_data, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'Histogram of {feature_of_interest}')
    plt.xlabel(feature_of_interest)
    plt.ylabel('Frequency')
    plt.show()

    feature_mean = np.mean(feature_data)
    feature_variance = np.var(feature_data)
    print(f"\nMean of {feature_of_interest}: {feature_mean}")
    print(f"Variance of {feature_of_interest}: {feature_variance}")

def calculate_minkowski_distance(df, feature1, feature2):
    data_feature1 = df[feature1].dropna()
    data_feature2 = df[feature2].dropna()

    r_values = np.arange(1, 11)
    distances = []

    for r in r_values:
        minkowski_distance = np.linalg.norm(data_feature1 - data_feature2, ord=r)
        distances.append(minkowski_distance)

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, distances, marker='o', linestyle='-', color='b')
    plt.title(f'Minkowski Distance between {feature1} and {feature2}')
    plt.xlabel('r')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

def split_data(df, class_label_column, selected_classes):
    df_selected_classes = df[df[class_label_column].isin(selected_classes)]
    label_encoder = LabelEncoder()
    df_selected_classes[class_label_column] = label_encoder.fit_transform(df_selected_classes[class_label_column])

    X = df_selected_classes.drop(columns=[class_label_column])
    y = df_selected_classes[class_label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def train_knn_classifier(X_train, y_train, k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh

def evaluate_classifier(neigh, X_test, y_test):
    y_pred = neigh.predict(X_test)
    accuracy = neigh.score(X_test, y_test)
    
    print(neigh.predict(X_test))
    return y_pred

def compare_knn_nn(X_train, y_train, X_test, y_test):
    k_values = list(range(1, 12))
    accuracy_scores_kNN = []
    accuracy_scores_NN = []

    for k in k_values:
        neigh_kNN = KNeighborsClassifier(n_neighbors=k)
        neigh_kNN.fit(X_train, y_train)
        accuracy_kNN = neigh_kNN.score(X_test, y_test)
        accuracy_scores_kNN.append(accuracy_kNN)

        neigh_NN = KNeighborsClassifier(n_neighbors=1)
        neigh_NN.fit(X_train, y_train)
        accuracy_NN = neigh_NN.score(X_test, y_test)
        accuracy_scores_NN.append(accuracy_NN)

    plt.plot(k_values, accuracy_scores_kNN, label='kNN (k = 3)')
    plt.plot(k_values, accuracy_scores_NN, label='NN (k = 1)')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: kNN vs NN')
    plt.legend()
    plt.show()

def evaluate_confusion_matrix_test(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    
def evaluate_confusion_matrix_train(y_train, y_test):
    conf_matrix = confusion_matrix(y_train,y_test)
    print("Confusion Matrix:")
    print(conf_matrix)

    precision = precision_score(y_train, y_test, average='weighted')
    recall = recall_score(y_train, y_test, average='weighted')
    f1 = f1_score(y_train, y_test, average='weighted')
    
def scatter_plot_with_knn_classification(X_train, y_train, X_test, y_test, k=3):
    neigh = train_knn_classifier(X_train, y_train, k=k)
    y_pred = evaluate_classifier(neigh, X_test, y_test)

    # Scatter plot of the test data output
    plt.scatter(X_test[y_pred == 0]['koi_score'], X_test[y_pred == 0]['koi_period'], color='blue', alpha=0.5, label='Predicted CANDIDATE')
    plt.scatter(X_test[y_pred == 1]['koi_score'], X_test[y_pred == 1]['koi_period'], color='red', alpha=0.5, label='Predicted CONFIRMED')

    # Scatter plot of the training data for reference
    plt.scatter(X_train[y_train == 0]['koi_score'], X_train[y_train == 0]['koi_period'], color='blue', label='Training CANDIDATE')
    plt.scatter(X_train[y_train == 1]['koi_score'], X_train[y_train == 1]['koi_period'], color='red', label='Training CONFIRMED')

    plt.xlabel('koi_score')
    plt.ylabel('koi_period')
    plt.title('Scatter Plot of Test Data with kNN Classification')
    plt.legend()
    plt.show()  

def hyperparameter_tuning(X_train, y_train):
    # Define the kNN classifier
    knn_classifier = KNeighborsClassifier()

    # Define the parameter grid for k
    param_dist = {'n_neighbors': np.arange(2, 21)}

    # Define the RandomizedSearchCV
    random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=5, random_state=42)

    # Fit the RandomizedSearchCV on the training data
    random_search.fit(X_train, y_train)

    # Print the best parameters and their corresponding accuracy
    print("Best Parameters: ", random_search.best_params_)
    print("Best Accuracy: ", random_search.best_score_)

    # Get the best kNN classifier
    best_knn_classifier = random_search.best_estimator_

    return best_knn_classifier   

def main():
    file_path = "KOI_dataset.csv"
    skip_rows = 86
    class_label_column = 'koi_disposition'
    selected_classes = ['CONFIRMED', 'CANDIDATE']
    
    # List of columns to drop
    columns_to_drop = ["kepler_name", "koi_comment", "koi_trans_mod", "koi_longp", "koi_model_dof", "koi_model_chisq", "koi_sage", "koi_ingress", "kepoi_name", "koi_vet_date", "koi_limbdark_mod", "koi_parm_prov", "koi_tce_delivname", "koi_sparprov", "koi_datalink_dvr", "koi_datalink_dvs", "koi_quarters"]

    # q1
    df = load_data(file_path, skip_rows)
    drop_columns(df, columns_to_drop)
    df_encoded = one_hot_encoding(df, class_label_column)

    # q2
    df_encoded = handle_missing_values(df_encoded)
    save_preprocessed_data(df_encoded, "preprocessed_data.csv")

    # q3
    calculate_mean_and_spread(df_encoded, class_label_column)

    # q4
    feature_of_interest = "koi_time0"
    plot_histogram_mean_variance(df_encoded, feature_of_interest)

    # q5
    X_train, X_test, y_train, y_test = split_data(df_encoded, class_label_column, selected_classes)

    # q6
    neigh = train_knn_classifier(X_train, y_train)
    print(f"Accuracy on the test set:")
    y_pred = evaluate_classifier(neigh, X_test, y_test)
    print(f"Accuracy on the train set:")
    y_pred_train=evaluate_classifier(neigh, X_train, y_train)

    # q7
    compare_knn_nn(X_train, y_train, X_test, y_test)

    # q8
    print("Evaluation of Confusion matrix, classfication report on test set :")
    evaluate_confusion_matrix_test(y_test, y_pred)
    
    ################################ASS-5###################################################
    #Q1
    print("Evaluation of Confusion matrix, classfication report on train set :")
    evaluate_confusion_matrix_test(y_train, y_pred_train)
 #inference of the model : 
 
 # we can see that out of the total 1418 instances, 954 are correctly classified by the model on the test 
 #set and 464 are wrongly classified.Whereas in case of the training set, out of the total 3307 instances,
 #2778 are correctly classified and 529 are wrongly classified. Here, we do not see the model to be overfitting to
 # the data as there is not a significant spike in the accuracy of training data as compared to the test set
 # and there is no significant gap in the performance metrics (precision, recall, f1-score) too between the train and 
 #test sets.
 #We can't say that the model is underfitting since the accuracies of both the train and test sets are not too low 
 #and the number of correct predictions outweigh the no. of wrong predictions so it wouldn't be apt to say that the 
 #model fails to capture the underlying patters in the data.
 #I would say that the model shows the characteristics of regular fit, as the performace metrics of train and test are 
 #both balanced (consistent) and the model performs reasonably well to both trained and unseen data ( without capturing the noise in the train set)
 
 #Q6
    scatter_plot_with_knn_classification(X_train, y_train, X_test, y_test, k=3)

 #Q7 - hyperparameter tuning to find best k value for the knn classifier
    best_knn_classifier = hyperparameter_tuning(X_train, y_train)
    best_knn_classifier.fit(X_train, y_train)
    y_pred = evaluate_classifier(best_knn_classifier, X_test, y_test)
if __name__ == "__main__":
    main()


