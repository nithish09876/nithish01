from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Load your trained model (make sure to adjust the path)
model = tf.keras.models.load_model('path_to_your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    # Process the image
    image_data = data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model

    # Make prediction
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)