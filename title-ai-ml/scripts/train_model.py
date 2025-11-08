from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from preprocess_data import load_datasets, preprocess_data
import pickle

def train_tensorflow_model(X_train, y_train, X_val, y_val, model=None):
    if model is None:
        # Define the model if not provided
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        # Re-compile the model with a new optimizer
        print("Recompiling the model with a new optimizer.")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Continue training the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model

if __name__ == "__main__":
    dataset_folder = './data'  # Adjust path as needed
    df = load_datasets(dataset_folder)
    X, vectorizer = preprocess_data(df)

    # Create dummy labels for training (replace with real labels if available)
    y = np.random.randint(0, 2, X.shape[0])

    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

    # Load the pre-trained model if it exists
    try:
        model = tf.keras.models.load_model('./model/title_verification_model.h5')
        print("Loaded pre-trained model.")
    except:
        print("No pre-trained model found. Training a new one.")
        model = None

    # Train the model (or fine-tune the pre-trained one)
    model = train_tensorflow_model(X_train, y_train, X_val, y_val, model=model)

    # Save the model after fine-tuning
    model.save('./model/title_verification_model.h5')
