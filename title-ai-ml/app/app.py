from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_model import TitleVerificationML
import numpy as np

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:8080"]}})

# Initialize the ML model and vectorizer
model_path = './model/title_verification_model.h5'
vectorizer_path = './model/tfidf_vectorizer.pkl'
disallowed_words = ['Police', 'Crime', 'Corruption', 'CBI', 'CID', 'Army']  # Add more disallowed words as needed
prefixes_suffixes = ['The', 'India', 'Samachar', 'News']  # Add more prefixes/suffixes as needed

ml_model = TitleVerificationML(model_path, vectorizer_path, disallowed_words, prefixes_suffixes)

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def check_title():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    data = request.get_json()
    new_title = data.get('title', '')

    # Validate title with the ML model
    is_valid, prediction_score = ml_model.validate_title(new_title)

    # Convert NumPy types to native Python types for JSON serialization
    if isinstance(is_valid, np.bool_):
        is_valid = bool(is_valid)
    if isinstance(prediction_score, np.float32):
        prediction_score = float(prediction_score)

    return jsonify({
        'is_valid': is_valid,
        'prediction_score': prediction_score
    })

if __name__ == '__main__':
    app.run(port=5001)
