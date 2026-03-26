from pathlib import Path
import pandas as pd


def main():
    p = Path('data/ISOT')
    if not p.exists():
        print('Directory data/ISOT not found. Please download files first.')
        return

    true_f = p / 'True.csv'
    fake_f = p / 'Fake.csv'
    if not true_f.exists() or not fake_f.exists():
        print('Missing True.csv or Fake.csv in data/ISOT')
        return

    df_true = pd.read_csv(true_f)
    df_fake = pd.read_csv(fake_f)

    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'

    def ensure_text(df):
        if 'text' in df.columns:
            return df
        if 'content' in df.columns:
            df['text'] = df['content']
            return df
        if 'title' in df.columns:
            df['text'] = df['title']
            return df
        raise ValueError('No text/content/title column found')

    df_true = ensure_text(df_true)
    df_fake = ensure_text(df_fake)

    # Prefer title + text when available
    def combine(df):
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'].fillna('') + '. ' + df['text'].fillna('')
        return df[['text', 'label']]

    out1 = combine(df_true)
    out2 = combine(df_fake)

    combined = pd.concat([out1, out2], ignore_index=True)
    combined = combined.dropna(subset=['text'])
    combined.to_csv('data/combined.csv', index=False)
    print('Wrote data/combined.csv with', len(combined), 'rows')


if __name__ == '__main__':
    main()
