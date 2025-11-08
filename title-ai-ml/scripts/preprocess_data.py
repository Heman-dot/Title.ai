import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_datasets(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path, engine='openpyxl')
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        raise ValueError("No valid datasets found in the folder.")

def preprocess_data(df):
    # Assuming 'Title Name' column contains the title information
    titles = df['Title Name'].fillna('').tolist()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(titles)
    return X, vectorizer

if __name__ == "__main__":
    dataset_folder = './data'  # Adjust path as needed
    df = load_datasets(dataset_folder)
    X, vectorizer = preprocess_data(df)

    # Save the TF-IDF vectorizer for later use
    with open('./model/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
