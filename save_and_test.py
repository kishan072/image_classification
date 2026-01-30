import tensorflow as tf
import numpy as np
from PIL import Image

# Recreate and compile the model
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model architecture
model.save('intel_image_classifier.h5')
print("Model saved successfully!")

# Test prediction
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
test_image = 'seg_test/seg_test/buildings/20057.jpg'

img = Image.open(test_image)
img = img.resize((150, 150))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = predictions[0][np.argmax(predictions[0])]

print(f"Actual: buildings")
print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print("Classification complete!")