import tensorflow as tf
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TitleVerificationML:
    def __init__(self, model_path, vectorizer_path, disallowed_words, prefixes_suffixes):
        # Load the TensorFlow model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Disallowed words and prefixes/suffixes
        self.disallowed_words = set(disallowed_words)
        self.prefixes_suffixes = set(prefixes_suffixes)

    def preprocess_title(self, title):
        """Transform title into TF-IDF representation."""
        title_tfidf = self.vectorizer.transform([title]).toarray()
        return title_tfidf

    def check_disallowed_words(self, title):
        """Check if the title contains any disallowed words."""
        title_words = set(re.findall(r'\w+', title.lower()))
        return not bool(title_words & self.disallowed_words)

    def check_prefix_suffix(self, title):
        """Check if the title contains disallowed prefixes or suffixes."""
        lower_title = title.lower()
        for prefix in self.prefixes_suffixes:
            if lower_title.startswith(prefix) or lower_title.endswith(prefix):
                return False
        return True

    def validate_title(self, new_title):
        """Validate the new title based on multiple criteria."""
        # Check disallowed words
        if not self.check_disallowed_words(new_title):
            return False, "Title contains disallowed words"

        # Check prefixes/suffixes
        if not self.check_prefix_suffix(new_title):
            return False, "Title contains disallowed prefixes or suffixes"
        
        # TF-IDF based similarity check - This part is not needed as there are no existing titles to compare against
        # Use the ML model to predict validity
        new_title_tfidf = self.preprocess_title(new_title)
        prediction = self.model.predict(new_title_tfidf)
        is_valid = prediction[0][0] > 0.8
        return is_valid, prediction[0][0]
