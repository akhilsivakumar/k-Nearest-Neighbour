# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
irisData = load_iris()

# Print information about the Iris dataset
print("Iris Dataset Information:")
print(irisData)

# Extract features and target variable
X = irisData.data
y = irisData.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-Nearest Neighbors classifier with 7 neighbors
knn = KNeighborsClassifier(n_neighbors=7)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Evaluate the accuracy of the classifier on the testing data
accuracy = knn.score(X_test, y_test)
print("Accuracy of k-Nearest Neighbors Classifier:", accuracy)

# Visualize the relationship between the number of neighbors and accuracy
neighbors = range(1, 21)  # Range of neighbor values to test
test_accuracy = []  # List to store testing accuracy values
train_accuracy = []  # List to store training accuracy values

# Iterate over different values of neighbors and train the classifier
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

# Plot the relationship between number of neighbors and accuracy
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('k-Nearest Neighbors Classifier Accuracy')
plt.show()
