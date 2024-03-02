from math import sqrt#to perform math functions

def findEuclidean(vec1,vec2):
    euclidean=0
    #finding l2 norm of the given vectors
    for i in range(len(vec2)):
            euclidean= euclidean + (vec2[i]-vec1[i])**2 #finding sum of squares of the difference of the vector elements
    return sqrt(euclidean) #or **0.5
    
def findManhattan(vec1,vec2):
    manhattan=0
    #absolute sum of the differences btw the x , y coordinates
    #l1 norm
    for j in range(len(vec2)):
            manhattan= manhattan+ abs(vec1[j]-vec2[j])#finding sum of absolute diffrences of vector elements
    return manhattan

def encodeLabel(categ):
#performing label encoding on the list of categories : categ
    label_mapping = {}  # Create an empty dictionary to store label mappings
    unique_labels = set(categ)  # Get unique labels in the list

    for i, label in enumerate(unique_labels):
        label_mapping[label] = i  # Assigning incremental numerical labels to unique categories

    # Replacing original labels with numerical labels in the list
    encoded_labels = [label_mapping[label] for label in categ]

    return encoded_labels

def encodeOneHot(categ):
    #performing one-hot encdoing on the list of categories : categ
    unique_labels=set(categ)
    #creating an array of columns = unique_labels and rows= no.of categories
    encoded_labels=[[0 for i in range(len(unique_labels))] for i in range(len(categ))]
    label_mapping={}
    for i,label in enumerate(unique_labels):
        label_mapping[label]=i #assigning index values to each label
    for i in range(len(categ)):
        for j in range(len(unique_labels)):
            if j==label_mapping[categ[i]]: #if the value of index of labels matches with the value of the category
                encoded_labels[i][j]=1
    return encoded_labels
    

def main():
    length1= int(input("Enter the length for vector 1:"))
    length2= int(input("Enter the length for vector 2:"))
    if length1 != length2:
        print("The lengths are not equal")
    else:
        print("Enter the elements of vector 1:")
        vec1= [int(input()) for i in range(length1)]
        print(vec1)
        print("Enter the elements of vector 2:")
        vec2= [int(input()) for j in range(length1)]
        print(vec2)
        print("The Euclidean distance between the 2 vectors is :",findEuclidean(vec1,vec2))
        print("The Manhattan distance between the 2 vectors is :",findManhattan(vec1,vec2))
    

    #categorical_data = ['red', 'blue', 'green', 'red', 'green']
    numberOfCateg=int(input("Enter the no. of categories: "))
    listOfCateg=[]
    for i in range((numberOfCateg)):
        item= input(f"Enter item {i} \t")
        listOfCateg.append(item)
        
        
    print(encodeLabel(listOfCateg))  
    print(encodeOneHot(listOfCateg))
    print("The label-encoded values for the given categorical data is as follows: ",encodeLabel(listOfCateg))  
    print("The One-hot encoded values for the given categorical data is as follows: ", encodeOneHot(listOfCateg))
        
if __name__ == "__main__":
    main()

    