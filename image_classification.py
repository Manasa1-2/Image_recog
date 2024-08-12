import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define paths (Update the paths according to your directory structure)
train_dir = './train'
validation_dir = './test'

# Create datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size=(150, 150),
    batch_size=32
)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Build the Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and Train the Model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=validation_dataset
)

# Evaluate the Model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(2)

print(f"Length of epochs_range: {len(epochs_range)}")
print(f"Length of acc: {len(acc)}")
print(f"Length of val_acc: {len(val_acc)}")
print(f"Length of loss: {len(loss)}")
print(f"Length of val_loss: {len(val_loss)}")

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_validation_metrics.png')  # Save the plot as an image file

# Save the Model (Optional)
model.save('image_classification.h5')

# Read and display an image
image_path = 'C:/Users/manas/Downloads/archive/dogs_vs_cats/train/cats/cat.12277.jpg'  # Replace with the actual path to your image
test_img = cv2.imread(image_path)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Display the image using OpenCV
cv2.imshow('Test Image', cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV display
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
