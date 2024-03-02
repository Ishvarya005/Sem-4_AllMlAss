import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(file_path, sheet_name):
    # Load the data from a specific sheet in the Excel file
    return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)

def calculate_equations(A, C):
    # Form equations: AX = C
    equations = []
    for i in range(A.shape[0]):
        equation = f"{A[i, 0]}x1 + {A[i, 1]}x2 + {A[i, 2]}x3 = {C[i]}"
        equations.append(equation)
    return equations

def calculate_rank(A):
    # Calculate the rank of matrix A
    return np.linalg.matrix_rank(A)

def calculate_cost_vector(A, C):
    # Calculate the pseudo-inverse of matrix A
    A_pseudo_inv = np.linalg.pinv(A)
    
    # Calculate the cost vector using the pseudo-inverse: Cost = A_pseudo_inv * C
    return np.dot(A_pseudo_inv, C)

def main():
    file_path = "D:/Sem-4/ML/Assignments/Ass3/Lab Session1 Data.xlsx"
    sheet_name = 'Purchase data'

    # Load data
    X = load_data(file_path, sheet_name)
    
    # Separate feature variables (A) and target variable (C)
    A_columns = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    C_column = 'Payment (Rs)'
    A = X[A_columns].values
    C = X[C_column].values.reshape(-1, 1)  # Reshape C to be a column vector

    # Impute missing values in A (if any)
    imputer = SimpleImputer(strategy='mean')
    A = imputer.fit_transform(A)

    # Print equations
    equations = calculate_equations(A, C)
    print("\nEquations:")
    for equation in equations:
        print(equation)

    # Calculate and print the rank of matrix A
    rank_A = calculate_rank(A)
    print("\nDimensionality of vector space of the data:", rank_A)

    # Print the number of vectors in A and C
    print("No. of vectors in A and C are as follows:", np.shape(A[1]), "and", np.shape(C[0]))

    # Print the rank of matrix A
    print("The rank of matrix A is:", rank_A)

    # Calculate and print the cost of each product
    cost_vector = calculate_cost_vector(A, C)
    cost_columns = ['Cost of Each Product']
    cost_df = pd.DataFrame(data=cost_vector, columns=cost_columns)
    print("\nCost of Each Product:")
    print(cost_df)

    # Calculate and print the model vector for predicting the cost of each product
    model_vector = calculate_cost_vector(A, C)  # Using the same function for simplicity
    model_columns = ['Model Vector X']
    model_df = pd.DataFrame(data=model_vector, columns=model_columns)
    print("\nModel Vector X for Predicting the Cost of Each Product:")
    print(model_df)

if __name__ == "__main__":
    main()
