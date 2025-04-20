from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('emotion_model.h5')

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Music recommendations
music_recommendations = {
    'Angry': ['Sweet Child O’ Mine - Guns N’ Roses', 'Break Stuff - Limp Bizkit', 'Killing in the Name - Rage Against the Machine'],
    'Disgust': ['Smells Like Teen Spirit - Nirvana', 'Creep - Radiohead', 'Bad Guy - Billie Eilish'],
    'Fear': ['Thriller - Michael Jackson', 'Psycho Therapy - The Ramones', 'Black Hole Sun - Soundgarden'],
    'Happy': ['Uptown Funk - Mark Ronson', 'Happy - Pharrell Williams', 'Dancing Queen - ABBA'],
    'Sad': ['Someone Like You - Adele', 'Fix You - Coldplay', 'My Heart Will Go On - Celine Dion'],
    'Surprise': ['Bohemian Rhapsody - Queen', 'Firework - Katy Perry', 'Sweet Caroline - Neil Diamond'],
    'Neutral': ['Shape of You - Ed Sheeran', 'Viva La Vida - Coldplay', 'Blinding Lights - The Weeknd']
}

# Preprocess image for model
def preprocess_image(image):
    if image is None:
        raise ValueError("Input image is None")
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from frontend
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: Failed to decode webcam image")
            return jsonify({'error': 'Failed to decode image'}), 500

        # Debug: Save input image
        cv2.imwrite('debug_input.jpg', image)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]

        # Debug: Log probabilities
        print(f"Webcam Prediction Probabilities: {prediction}")

        # Get music recommendations
        recommendations = music_recommendations[emotion]

        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)