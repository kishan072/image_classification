import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Recreate the model (same as in main script)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Load saved weights if available
try:
    model.load_weights('intel_image_classifier.h5')
    print("Loaded saved model weights")
except:
    print("No saved weights found. Please run the main training script first.")
    exit()

# Test image
test_image_path = 'seg_test/seg_test/buildings/20057.jpg'
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Load and preprocess image
img = Image.open(test_image_path)
img_resized = img.resize((150, 150))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_index]
confidence = predictions[0][predicted_class_index]

# Display
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title(f'Actual: buildings | Predicted: {predicted_class} | Confidence: {confidence:.2f}')
plt.axis('off')
plt.show()

print(f"Actual: buildings")
print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.4f}")