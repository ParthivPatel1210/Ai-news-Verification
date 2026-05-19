"""
train.py
========
Downloads and caches the BERT fake-news model locally (bert_model/).
Also keeps the legacy TF-IDF + SGD training path for reference.

Usage:
  python train.py            ← downloads BERT model (recommended)
  python train.py --tfidf    ← retrain TF-IDF model from data/combined.csv
"""

import argparse
from pathlib import Path


def download_bert():
    """Download sentence-transformers/all-MiniLM-L6-v2 and save to ./bert_model/"""
    ST_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    save_dir    = Path("bert_model")

    if save_dir.exists() and any(save_dir.iterdir()):
        # Quick check: does it contain a sentence-transformer model (has modules.json)?
        if (save_dir / "modules.json").exists():
            print(f"[BERT] Sentence transformer already cached at ./{save_dir}/ — skipping.")
            return
        else:
            # Old broken bert-tiny model — wipe and redownload
            import shutil
            print("[BERT] Removing old model cache (incompatible format)...")
            shutil.rmtree(save_dir)

    print(f"[BERT] Downloading '{ST_MODEL_ID}' (~22 MB) from HuggingFace Hub ...")
    print("       This model creates genuine BERT semantic embeddings.\n")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ST_MODEL_ID)
        save_dir.mkdir(exist_ok=True)
        model.save(str(save_dir))
        print(f"\n[BERT] ✓ Sentence transformer saved to ./{save_dir}/")
        print("[BERT] The server will now use semantic prototype classification.\n")

    except Exception as e:
        print(f"[BERT] Download failed: {e}")
        print("       Check your internet connection and try again.")
        raise


def train_tfidf():
    """Legacy TF-IDF + SGD training (kept for backward compatibility)."""
    from pathlib import Path
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import joblib

    data_path = Path('data/combined.csv')
    if not data_path.exists():
        data_path = Path('data/sample.csv')
    if not data_path.exists():
        print('Missing data/combined.csv or data/sample.csv.')
        return

    df = pd.read_csv(data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        print('CSV must have `text` and `label` columns.')
        return

    vectorizer = TfidfVectorizer(max_features=12000, stop_words='english', ngram_range=(1, 2))
    print('Fitting TF-IDF vectorizer …')
    X  = vectorizer.fit_transform(df['text'].astype(str))
    le = LabelEncoder()
    y  = le.fit_transform(df['label'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training SGDClassifier …')
    model = SGDClassifier(loss='log_loss', max_iter=2500,
                          penalty='l2', alpha=1e-4, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    joblib.dump(model,      'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(le,         'label_encoder.joblib')
    print(f'TF-IDF training done. Test accuracy: {acc:.4f}')
    print('Saved model.joblib / vectorizer.joblib / label_encoder.joblib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI NewsVerifier — model manager')
    parser.add_argument('--tfidf', action='store_true',
                        help='Retrain legacy TF-IDF model instead of downloading BERT')
    args = parser.parse_args()

    if args.tfidf:
        train_tfidf()
    else:
        download_bert()
