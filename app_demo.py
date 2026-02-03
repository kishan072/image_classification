try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    print("Flask is not installed. Please install it with: pip install flask")
    exit(1)

import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image):
    try:
        # Mock prediction for demo (replace with actual model when available)
        img = image.resize((150, 150))
        
        # Simple mock prediction based on image properties
        img_array = np.array(img)
        avg_color = np.mean(img_array, axis=(0,1))
        
        # Mock logic for demo
        if avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:  # More blue
            predicted_class = 'sea'
            confidence = 0.85
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:  # More green
            predicted_class = 'forest'
            confidence = 0.82
        else:
            predicted_class = 'buildings'
            confidence = 0.78
            
        # Create mock predictions for all classes
        predictions = np.random.rand(6)
        predictions = predictions / np.sum(predictions)  # Normalize
        
        # Set the predicted class to have higher probability
        pred_idx = class_names.index(predicted_class)
        predictions[pred_idx] = confidence
        predictions = predictions / np.sum(predictions)  # Re-normalize
        
        return predicted_class, confidence, predictions
        
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        print(f"Processing file: {file.filename}")
        
        # Read and process image
        image = Image.open(file.stream)
        print(f"Image mode: {image.mode}, Size: {image.size}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB")
        
        # Get prediction
        predicted_class, confidence, all_predictions = predict_image(image)
        print(f"Prediction: {predicted_class} with confidence: {confidence:.4f}")
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Prepare all class probabilities
        class_probabilities = {class_names[i]: float(all_predictions[i]) for i in range(len(class_names))}
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'accuracy_percentage': f"{confidence * 100:.2f}%",
            'image': img_str,
            'all_predictions': class_probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)