
import os
import numpy as np
from PIL import Image

# Directory paths
train_directory = '/home/pooja-sile/Desktop/2_folder/Training'
valid_directory = '/home/pooja-sile/Desktop/2_folder/Validation'
test_directory = '/home/pooja-sile/Desktop/2_folder/Testing'

# Number of epochs
epochs = 2

# Initialize perceptron weights and bias
num_features = None
weights = None
bias = None
learning_rate = 0.1

# Read training data to initialize weights and bias
X_train = []
y_train = []
for filename in os.listdir(train_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(train_directory, filename)
        img = Image.open(file_path)
        img = img.resize((100, 100)).convert('L')  # Resize and convert to grayscale ('L' mode)
        img_array = np.array(img)
        X_train.append(img_array.flatten())  # Flatten image into 1D array
        # Assuming the label is encoded in the filename or directory structure
        label = 1 if 'class1' in filename else 0  # Adjust based on your data
        y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize weights and bias
num_features = X_train.shape[1]
weights = np.zeros(num_features)
bias = 0.0

# Training and evaluation in a single loop
for epoch in range(epochs):
    # Training loop
    correct_train = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y_true = y_train[i]
        
        # Predict
        linear_output = np.dot(weights, x) + bias
        y_pred = 1 if linear_output >= 0 else 0
        
        # Update weights and bias if prediction is incorrect
        if y_true != y_pred:
            if y_true == 1:
                weights += learning_rate * x
                bias += learning_rate
            else:
                weights -= learning_rate * x
                bias -= learning_rate
        
        # Calculate training accuracy
        if y_true == y_pred:
            correct_train += 1
    train_accuracy = correct_train / len(X_train)
    print(f"Epoch {epoch + 1}: Train Accuracy = {train_accuracy:.2%}")
    
    # Validation accuracy
    X_valid = []
    y_valid = []
    for filename in os.listdir(valid_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(valid_directory, filename)
            img = Image.open(file_path)
            img = img.resize((100, 100)).convert('L')  # Resize and convert to grayscale ('L' mode)
            img_array = np.array(img)
            X_valid.append(img_array.flatten())  # Flatten image into 1D array
            # Assuming the label is encoded in the filename or directory structure
            label = 1 if 'class1' in filename else 0  # Adjust based on your data
            y_valid.append(label)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    
    correct_valid = 0
    for i in range(len(X_valid)):
        x = X_valid[i]
        y_true = y_valid[i]
        linear_output = np.dot(weights, x) + bias
        y_pred = 1 if linear_output >= 0 else 0
        if y_true == y_pred:
            correct_valid += 1
    validation_accuracy = correct_valid / len(X_valid)
    print(f"Epoch {epoch + 1}: Validation Accuracy = {validation_accuracy:.2%}")
    
    # Test accuracy
    X_test = []
    y_test = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(test_directory, filename)
            img = Image.open(file_path)
            img = img.resize((100, 100)).convert('L')  # Resize and convert to grayscale ('L' mode)
            img_array = np.array(img)
            X_test.append(img_array.flatten())  # Flatten image into 1D array
            # Assuming the label is encoded in the filename or directory structure
            label = 1 if 'class1' in filename else 0  # Adjust based on your data
            y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    correct_test = 0
    for i in range(len(X_test)):
        x = X_test[i]
        y_true = y_test[i]
        linear_output = np.dot(weights, x) + bias
        y_pred = 1 if linear_output >= 0 else 0
        if y_true == y_pred:
            correct_test += 1
    test_accuracy = correct_test / len(X_test)
    print(f"Epoch {epoch + 1}: Test Accuracy = {test_accuracy:.2%}")
