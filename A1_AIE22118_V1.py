#initialisation
vowels = ['a', 'e', 'i', 'o', 'u']
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']

def VowConsCount(s):
    count_vowels = 0
    count_consonants = 0
    for char in s:
        if char in consonants:
            count_consonants += 1
        if char in vowels:
            count_vowels += 1
    return count_vowels, count_consonants

def commonElem(list1, list2):
    common_elements = [num for num in list1 if num in list2] #using list comprehension to check if both conditions are satisfied
    return common_elements

def matrixMult(mat1, mat2):
    rows_mat1, cols_mat1 = len(mat1), len(mat1[0])
    rows_mat2, cols_mat2 = len(mat2), len(mat2[0])

    if cols_mat1 != rows_mat2:
        return "Error: Matrices A and B are not multiplicable"
    
    result_matrix = [[0 for _ in range(cols_mat2)] for _ in range(rows_mat1)]
    for i in range(rows_mat1):
        for j in range(cols_mat2):
            for k in range(rows_mat2):
                result_matrix[i][j] += mat1[i][k] * mat2[k][j] #finding sum of dot products of row*col
    return result_matrix

def transpose(mat):
    transposed = [[0 for _ in range(len(mat))] for _ in range(len(mat[0]))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            transposed[j][i] = mat[i][j]
    return transposed

def main():
    while True:
        #based on the questions in the assignment: the functions are arranged
        print("1. Count vowels and consonants in a string")
        print("2. Find common elements in two lists")
        print("3. Multiply two matrices")
        print("4. Transpose a matrix")
        print("Enter 'stop' to exit")

        choice = input("Enter the number corresponding to the operation you want to perform: ")

        if choice.lower() == 'stop':
            break #to exit the code 
        
        choice = int(choice)

        if choice == 1:
            string_input = input("Enter the string: ")
            count_vowels, count_consonants = VowConsCount(string_input)
            print("No. of vowels:", count_vowels)
            print("No. of consonants:", count_consonants)
        elif choice == 2:
            list1 = list(map(int, input("Enter the first list of integers separated by spaces: ").split()))
            list2 = list(map(int, input("Enter the second list of integers separated by spaces: ").split()))
            common_elements = commonElem(list1, list2)
            print("Common Elements:", common_elements)
        elif choice == 3:
            rows_mat1 = int(input("Enter the number of rows for matrix 1: "))
            cols_mat1 = int(input("Enter the number of columns for matrix 1: "))
            rows_mat2 = int(input("Enter the number of rows for matrix 2: "))
            cols_mat2 = int(input("Enter the number of columns for matrix 2: "))

            mat1 = [[int(input(f"Enter element for matrix 1 at position ({i + 1}, {j + 1}): ")) for j in range(cols_mat1)] for i in range(rows_mat1)]
            mat2 = [[int(input(f"Enter element for matrix 2 at position ({i + 1}, {j + 1}): ")) for j in range(cols_mat2)] for i in range(rows_mat2)]

            result_matrix = matrixMult(mat1, mat2)
            print(result_matrix)
        elif choice == 4:
            rows_mat = int(input("Enter the number of rows for the matrix: "))
            cols_mat = int(input("Enter the number of columns for the matrix: "))

            mat = [[int(input(f"Enter element at position ({i + 1}, {j + 1}): ")) for j in range(cols_mat)] for i in range(rows_mat)]

            transposed = transpose(mat)
            print(transposed)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
