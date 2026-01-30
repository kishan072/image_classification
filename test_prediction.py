import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('intel_image_classifier.h5')

# Test image path
test_image_path = 'seg_test/seg_test/buildings/20057.jpg'

def predict_and_display(image_path, model):
    # Class names
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    # Get actual class from folder name
    actual_class = os.path.basename(os.path.dirname(image_path))
    
    # Display results
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f'Actual: {actual_class} | Predicted: {predicted_class} | Confidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()
    
    print(f"Actual class: {actual_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Correct prediction: {actual_class == predicted_class}")
    
    return predicted_class, confidence

# Test the model
predict_and_display(test_image_path, model)