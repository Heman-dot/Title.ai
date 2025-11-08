from ml_model import TitleVerificationML
import numpy as np;
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(title_verifier, X_test, y_test):
    """
    Evaluates the performance of the TitleVerificationML model on a test dataset.
    
    Parameters:
    - title_verifier: an instance of TitleVerificationML class
    - X_test: list of test titles
    - y_test: list of true labels (0 or 1)

    Returns:
    Prints the calculated performance metrics.
    """

    # Store predictions and true labels
    y_pred = []
    y_prob = []

    for title in X_test:
        # Get prediction probability from the model
        is_valid, prob = title_verifier.validate_title(title)
        y_prob.append(prob)
        y_pred.append(1 if prob > 0.8 else 0)

    # Convert to numpy arrays for metric calculation
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_test = np.array(y_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # Precision
    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision:.4f}')

    # Recall
    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall:.4f}')

    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f'F1-Score: {f1:.4f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f'ROC-AUC: {roc_auc:.4f}')

# Assuming you have the test titles and labels
X_test = ["THE KING TIMES", "SANJHA TIMES", "TIMES OF AYURVED"]  # List of test titles
y_test = [1, 0, 1]  # Corresponding true labels for test titles

# Initialize your TitleVerificationML instance
model_path = "./model/title_verification_model.h5"
vectorizer_path = "./model/tfidf_vectorizer.pkl"
disallowed_words = ["Police", "Army"]
prefixes_suffixes = ["The", "News"]

title_verifier = TitleVerificationML(model_path, vectorizer_path, disallowed_words, prefixes_suffixes)

# Evaluate the model
evaluate_model(title_verifier, X_test, y_test)
