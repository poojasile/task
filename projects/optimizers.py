import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Define constants
base_directory = '/home/pooja-sile/Desktop/2_folder/'
train_directory = os.path.join(base_directory, 'Training')
valid_directory = os.path.join(base_directory, 'Validation')
test_directory = os.path.join(base_directory, 'Testing')
image_width, image_height = 28, 28  # Image dimensions
batch_size = 32
epochs = 50

# Initialize lists to hold data and labels
train_data, train_labels = [], []
valid_data, valid_labels = [], []
test_data, test_labels = [], []

# Load and preprocess images for training
directory = train_directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(directory, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_width, image_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        train_data.append(img_array)
        train_labels.append(0)  # Assuming all images are of the same class (e.g., class 0)

# Load and preprocess images for validation
directory = valid_directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(directory, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_width, image_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        valid_data.append(img_array)
        valid_labels.append(0)  # Assuming all images are of the same class (e.g., class 0)

# Load and preprocess images for testing
directory = test_directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(directory, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_width, image_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        test_data.append(img_array)
        test_labels.append(0)  # Assuming all images are of the same class (e.g., class 0)

# Convert lists to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Print number of images found
print(f"Found {len(train_data)} images for training.")
print(f"Found {len(valid_data)} images for validation.")
print(f"Found {len(test_data)} images for testing.")

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generator for training
train_generator = train_datagen.flow(
    x=train_data,
    y=train_labels,
    batch_size=batch_size,
    shuffle=True
)

valid_generator = valid_datagen.flow(
    x=valid_data,
    y=valid_labels,
    batch_size=batch_size,
    shuffle=False  # No need to shuffle validation data
)

# List of optimizers to try
optimizers = ['sgd', 'adam', 'rmsprop']

# Dictionary to store results
results = {}

# Iterate over each optimizer
for optimizer in optimizers:
    print(f"\nTraining with {optimizer} optimizer:")
    
    # Define model with current optimizer
    model = Sequential([
        Flatten(input_shape=(image_width, image_height, 3)),  # input layer
        Dense(128, activation='relu'),  # hidden layer
        BatchNormalization(),  # batch normalization layer
        Dropout(0.5),  # dropout layer
        Dense(64, activation='relu'),  # additional hidden layer
        BatchNormalization(),  # batch normalization layer
        Dropout(0.3),  # dropout layer
        Dense(1, activation='sigmoid')  # output layer, sigmoid for binary classification
    ])
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[early_stopping]
    )

    # Evaluate the model on training data
    train_loss, train_accuracy = model.evaluate(train_generator, steps=len(train_generator))

    # Evaluate the model on validation data
    valid_loss, valid_accuracy = model.evaluate(valid_generator, steps=len(valid_generator))

    # Evaluate the model on testing data
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    # Print and store results
    results[optimizer] = {
        'train_accuracy': train_accuracy,
        'valid_accuracy': valid_accuracy,
        'test_accuracy': test_accuracy
    }

    print(f'Training accuracy: {train_accuracy}')
    print(f'Validation accuracy: {valid_accuracy}')
    print(f'Testing accuracy with {optimizer} optimizer: {test_accuracy}')

# Print final comparison of results
print("\nFinal comparison of results:")
for optimizer, result in results.items():
    print(f"\nOptimizer: {optimizer}")
    print(f"Training accuracy: {result['train_accuracy']}")
    print(f"Validation accuracy: {result['valid_accuracy']}")
    print(f"Testing accuracy: {result['test_accuracy']}")