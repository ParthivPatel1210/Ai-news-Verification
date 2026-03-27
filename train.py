from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


def main():
    # Prefer a larger combined dataset if available
    data_path = Path('data/combined.csv')
    if not data_path.exists():
        data_path = Path('data/sample.csv')
    if not data_path.exists():
        print('Missing data/combined.csv or data/sample.csv. Add a labeled dataset and retry.')
        return

    df = pd.read_csv(data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        print('CSV must contain `text` and `label` columns.')
        return

    from sklearn.model_selection import train_test_split
    # Drastically reduced feature space to prevent 512MB RAM Out-Of-Memory limit on Render Free Tier
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 1))
    print('Fitting vectorizer...')
    X = vectorizer.fit_transform(df['text'].astype(str))
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training Logistic Regression model...')
    model = LogisticRegression(max_iter=2000, C=5.0)
    model.fit(X_train, y_train)

    # Evaluate quickly
    acc = model.score(X_test, y_test)

    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(le, 'label_encoder.joblib')

    print(f'Training complete. Accuracy on held-out test set: {acc:.4f}')
    print('Saved model.joblib, vectorizer.joblib, label_encoder.joblib')


if __name__ == '__main__':
    main()
