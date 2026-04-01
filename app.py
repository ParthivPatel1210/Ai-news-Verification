from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import joblib
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Prediction, Vote
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import ssl
from datetime import datetime
from bs4 import BeautifulSoup
import urllib.parse
import re
import numpy as np
from duckduckgo_search import DDGS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-in-production'
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT, 'data')
os.makedirs(data_dir, exist_ok=True)

# Securely bind to Cloud PostgreSQL if provided (Render), otherwise fallback to local SQLite
db_url = os.environ.get('DATABASE_URL')
# SQLAlchemy 1.4+ requires 'postgresql://' instead of 'postgres://'
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or f"sqlite:///{os.path.join(data_dir, 'users.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# init database and login manager
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


MODEL_PATH = Path('model.joblib')
VEC_PATH = Path('vectorizer.joblib')
LE_PATH = Path('label_encoder.joblib')

model = None
vectorizer = None
le = None


def load_artifacts():
    global model, vectorizer, le
    if MODEL_PATH.exists() and VEC_PATH.exists() and LE_PATH.exists():
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        le = joblib.load(LE_PATH)
        print('Loaded model artifacts.')
    else:
        print('Model artifacts not found. Run `python train.py` first.')


from sqlalchemy import text

with app.app_context():
    db.create_all()
    
    # Dynamically inject the new Base64 column into deployed Postgres/SQLite instances
    try:
        db.session.execute(text('ALTER TABLE users ADD COLUMN avatar_base64 TEXT'))
        db.session.commit()
        print("Successfully migrated avatar_base64 column.")
    except Exception:
        db.session.rollback()
    
    # Live schema migration for new accounts table columns
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN theme_preference VARCHAR(10) DEFAULT 'dark';"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN city VARCHAR(100);"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN country VARCHAR(100);"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN birthdate DATE;"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN gender VARCHAR(20);"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    load_artifacts()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api')
def api_docs():
    return render_template('api.html')


# --- Helper Functions for Advanced Features --- #
def scrape_article(url):
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Scraping error: {e}")
        return None

def get_credibility_score(url):
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
            
        trusted = {
            'bbc.com': 9.5, 'bbc.co.uk': 9.5, 'reuters.com': 9.8, 'apnews.com': 9.8,
            'npr.org': 9.0, 'nytimes.com': 8.5, 'wsj.com': 8.5, 'washingtonpost.com': 8.5,
            'theguardian.com': 8.0, 'bloomberg.com': 9.0, 'ft.com': 9.0, 'economist.com': 9.0,
            'cnn.com': 7.5, 'nbcnews.com': 8.0, 'cbsnews.com': 8.5, 'abcnews.go.com': 8.5,
            'foxnews.com': 5.5, 'nypost.com': 5.0, 'usatoday.com': 8.0, 'politico.com': 8.0,
            'aljazeera.com': 7.5, 'theatlantic.com': 8.0, 'time.com': 8.0, 'snopes.com': 9.5
        }
        untrusted = {
            'infowars.com': 1.0, 'breitbart.com': 3.0, 'theonion.com': 1.0,
            'babylonbee.com': 1.0, 'naturalnews.com': 1.0, 'thegatewaypundit.com': 2.0,
            'wnd.com': 3.0, 'newsmax.com': 4.0, 'oann.com': 3.0, 'sputniknews.com': 2.0,
            'rt.com': 2.0, 'dailywire.com': 4.5
        }
        
        if domain in trusted:
            return trusted[domain], "Known Trusted Journalistic Source"
        if domain in untrusted:
            return untrusted[domain], "Known Unreliable or Satire Source"
            
        # If unknown, use DDGS to heuristically check reputation
        query = f'"{domain}" news reliability bias'
        results = DDGS().text(query, max_results=3)
        
        score = 5.0 # Baseline for unknown
        reason = "Unknown Blog/Source - Average Reliability"
        
        if results:
            snippets = " ".join([r.get('body', '').lower() for r in results])
            if 'satire' in snippets or 'satirical' in snippets:
                score -= 3.0
                reason = "Unknown Source - Flagged as possible satire online."
            elif 'conspiracy' in snippets or 'fake news' in snippets or 'unreliable' in snippets:
                score -= 2.5
                reason = "Unknown Source - Flagged as potentially unreliable online."
            elif 'award-winning' in snippets or 'pulitzer' in snippets or 'reputable' in snippets or 'reliable' in snippets:
                score += 2.0
                reason = "Unknown Source - Contains online mentions of good journalistic reputation."
                
        score = max(1.0, min(score, 10.0))
        return round(score, 1), reason
        
    except Exception as e:
        print(f"Credibility API Error: {e}")
        return 3.0, "Unknown Blog - Error checking reputation"

def search_web_verification(text):
    try:
        # Construct query from first few words to capture the core claim/headline
        words = text.split()
        if len(words) > 10:
            headline = ' '.join(words[:10]) + " news"
        else:
            headline = ' '.join(words) + " news"
            
        results = DDGS().text(headline, max_results=3)
        # Format results
        out = []
        for r in results:
            out.append({
                'title': r.get('title', ''),
                'href': r.get('href', ''),
                'body': r.get('body', '')[:150] + '...'
            })
        return out
    except Exception as e:
        print(f"Web Search Error: {e}")
        return None

def extract_model_highlights(text, vectorizer, model, le, is_fake):
    try:
        if not hasattr(model, 'coef_'):
            return []
        
        classes = list(le.classes_)
        fake_idx = 0 if str(classes[0]).upper() == 'FAKE' else 1
        
        coefs = model.coef_[0]
        if fake_idx == 0:
            coefs = -coefs # Invert so positive values strongly indicate FAKE
            
        vec = vectorizer.transform([text])
        contributions = vec.toarray()[0] * coefs
        
        top_indices = np.argsort(contributions)[-6:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        
        # Filter for positive FAKE contributions only
        suspicious = [str(feature_names[i]) for i in top_indices if contributions[i] > 0]
        return suspicious
    except Exception as e:
        print(f"Extraction Error: {e}")
        return []

def generate_explanation_and_highlights(text, vectorizer, model, le, is_fake):
    suspicious_features = extract_model_highlights(text, vectorizer, model, le, is_fake)
    
    if is_fake:
        if suspicious_features:
            reason = f"The AI analyzed the linguistic structure and heavily weighted the usage of words like '{suspicious_features[0]}' and '{suspicious_features[1] if len(suspicious_features)>1 else 'these terms'}' as strong indicators of fabricated or highly biased content."
        else:
            reason = "The text exhibits general sensationalist linguistic patterns often associated with fabricated content."
    else:
        reason = "The text structure and vocabulary align with standard, objective journalistic reporting patterns. Very few sensationalist markers were detected."
        
    return reason, ','.join(suspicious_features)


@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer, le
    data = request.get_json() or request.form
    text = data.get('text', '').strip()
    url = data.get('url', '').strip()
    
    source_domain = None
    credibility = None
    credibility_reason = None
    
    if url and not text:
        scraped_text = scrape_article(url)
        if not scraped_text:
            return jsonify({'error': 'Failed to extract text from the provided URL.'}), 400
        text = scraped_text
        source_domain = urllib.parse.urlparse(url).netloc
        if source_domain.startswith('www.'):
            source_domain = source_domain[4:]
        
        c_val, c_reason = get_credibility_score(url)
        credibility = c_val
        credibility_reason = c_reason

    if not text:
        return jsonify({'error': 'No text or URL provided'}), 400
        
    if model is None:
        return jsonify({'error': 'Model not found. Run train.py first.'}), 500

    X = vectorizer.transform([text])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        pred_idx = int(proba.argmax())
        prob = float(proba[pred_idx])
    else:
        pred_idx = int(model.predict(X)[0])
        prob = 1.0

    label = le.inverse_transform([pred_idx])[0]
    
    # Base ML Evaluation
    ml_is_fake = (label.upper() == 'FAKE')
    ml_real_prob = prob if not ml_is_fake else (1.0 - prob)
    
    # Live Web Search
    web_results = search_web_verification(text)
    
    web_score = 0.5
    web_explanation = ""
    
    if web_results is not None:
        if len(web_results) > 0:
            valid_sources = 0
            avg_credibility = 0
            for res in web_results:
                score, _ = get_credibility_score(res['href'])
                avg_credibility += score
                valid_sources += 1
                
            if valid_sources > 0:
                avg_cred = avg_credibility / valid_sources
                if avg_cred >= 7.0:
                    web_score = 0.95
                    web_explanation = "We verified this claim against live data. It is currently being actively reported by highly credible journalistic sources."
                elif avg_cred >= 5.0:
                    web_score = 0.75
                    web_explanation = "We verified this claim online. It is currently being mentioned by various sources across the web."
                else:
                    web_score = 0.35
                    web_explanation = "While we found articles discussing this, they originated from historically unreliable or heavily biased sources, reducing credibility."
        else:
            web_score = 0.15
            web_explanation = "A live scan of the internet yielded zero results for this claim from any recognizable publisher, heavily increasing the likelihood that it is entirely fabricated."
            
        final_real_prob = (ml_real_prob * 0.40) + (web_score * 0.60)
    else:
        final_real_prob = ml_real_prob
        web_explanation = "Due to network conditions, live web verification was unavailable. This prediction relies entirely on linguistics."
        web_results = [] # ensure empty array rather than null for safety
        
    if final_real_prob > 0.5:
        final_label = "REAL"
        final_prob = final_real_prob
        final_is_fake = False
    else:
        final_label = "FAKE"
        final_prob = 1.0 - final_real_prob
        final_is_fake = True
        
    # Generate Advanced feature explanations based on original ML thought
    explanation, suspicious_words = generate_explanation_and_highlights(text, vectorizer, model, le, ml_is_fake)
    
    # Merge Explanations
    explanation = f"<strong>🤖 ML Linguistic Filter:</strong> {explanation} <br><br><strong>🌐 Live Web Ensemble:</strong> {web_explanation}"
    
    # Update local variables for DB and JSON return
    label = final_label
    prob = final_prob
    is_fake = final_is_fake
    
    # If using pure text input without URL, attempts to guess the source from the DDGS top search result!
    if not url and web_results and not source_domain:
        guessed_url = web_results[0].get('href', '')
        if guessed_url:
            source_domain = urllib.parse.urlparse(guessed_url).netloc
            if source_domain.startswith('www.'):
                source_domain = source_domain[4:]
            c_val, c_reason = get_credibility_score(guessed_url)
            credibility = c_val
            credibility_reason = f"Source guessed from text match: {c_reason}"

    prediction_id = None
    # Save history if user logged in
    try:
        if current_user and current_user.is_authenticated:
            p = Prediction(
                user_id=int(current_user.get_id()), 
                text=text[:1000] + ('...' if len(text)>1000 else ''), # truncate for DB safe
                prediction=label, 
                probability=prob, 
                timestamp=datetime.utcnow(),
                url=url if url else None,
                source_credibility=credibility,
                explanation=explanation,
                suspicious_words=suspicious_words
            )
            db.session.add(p)
            db.session.commit()
            prediction_id = p.id
    except Exception as e:
        print(f"DB Error: {e}")
        pass
        
    return jsonify({
        'prediction': label, 
        'probability': prob,
        'scraped_text': text if url else None,
        'source_domain': source_domain,
        'source_credibility': credibility,
        'credibility_reason': credibility_reason,
        'explanation': explanation,
        'suspicious_words': suspicious_words.split(',') if suspicious_words else [],
        'web_results': web_results,
        'prediction_id': prediction_id
    })


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username').lower().strip() if request.form.get('username') else None
        email = request.form.get('email').lower().strip() if request.form.get('email') else None
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password or not confirm_password:
            flash('Fill all fields', 'danger')
            return redirect(url_for('signup'))
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))
            
        if User.query.filter_by(username=username).first():
            flash('Username is already taken.', 'danger')
            return redirect(url_for('signup'))
            
        if User.query.filter_by(email=email).first():
            flash('Email is already registered. Please login.', 'danger')
            return redirect(url_for('signup'))

        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').lower().strip() if request.form.get('username') else None
        password = request.form.get('password')
        user = User.query.filter((User.username == username) | (User.email == username)).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        bio = request.form.get('bio')
        city = request.form.get('city')
        country = request.form.get('country')
        gender = request.form.get('gender')
        birthdate_str = request.form.get('birthdate')
        profile_picture = request.form.get('profile_picture')
        
        # Base64 Persistent Fast-Avatar Processor
        avatar_file = request.files.get('profile_picture_file')
        if avatar_file and avatar_file.filename != '':
            try:
                from PIL import Image
                import io
                import base64
                img = Image.open(avatar_file.stream).convert('RGB')
                img.thumbnail((250, 250)) # Compress for DB storage performance
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                current_user.avatar_base64 = img_str
                # Clear profile_picture URL since local file was uploaded
                current_user.profile_picture = None
            except Exception as e:
                flash(f"Error processing image: {str(e)}")

        current_user.bio = bio
        current_user.city = city
        current_user.country = country
        current_user.gender = gender
        
        if birthdate_str:
            try:
                current_user.birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d').date()
            except ValueError:
                pass
                
        if profile_picture is not None:
            # If the user submitted an empty URL box, switch to UI Avatars default
            if profile_picture.strip() == '':
                current_user.profile_picture = f'https://ui-avatars.com/api/?name={current_user.username}&background=10b981&color=fff'
            else:
                current_user.profile_picture = profile_picture
            
        db.session.commit()
        flash('Profile settings updated successfully!', 'success')
        return redirect(url_for('dashboard'))
        
    history = Prediction.query.filter_by(user_id=int(current_user.get_id())).order_by(Prediction.timestamp.desc()).limit(200).all()
    return render_template('dashboard.html', history=history)


@app.route('/settings/theme', methods=['POST'])
@login_required
def update_theme():
    theme = request.form.get('theme')
    if theme in ['light', 'dark']:
        current_user.theme_preference = theme
        db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/settings/username', methods=['POST'])
@login_required
def update_username():
    new_username = request.form.get('new_username')
    password = request.form.get('password')
    if not check_password_hash(current_user.password_hash, password):
        flash('Incorrect current password.', 'danger')
        return redirect(url_for('dashboard'))
    if User.query.filter_by(username=new_username).first():
        flash('Username is already taken by another user.', 'danger')
        return redirect(url_for('dashboard'))
    
    current_user.username = new_username
    db.session.commit()
    flash('Username updated successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/settings/password', methods=['POST'])
@login_required
def update_password():
    old_password = request.form.get('old_password')
    new_password = request.form.get('new_password')
    
    if not check_password_hash(current_user.password_hash, old_password):
        flash('Incorrect current password.', 'danger')
        return redirect(url_for('dashboard'))
        
    current_user.password_hash = generate_password_hash(new_password)
    db.session.commit()
    flash('Password securely updated!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/community')
def community():
    public_posts = Prediction.query.filter_by(is_public=True).order_by(Prediction.timestamp.desc()).limit(100).all()
    return render_template('community.html', posts=public_posts)


@app.route('/publish/<int:pid>', methods=['POST'])
@login_required
def publish_prediction(pid):
    p = Prediction.query.get(pid)
    if not p or p.user_id != int(current_user.get_id()):
        flash('Not authorized or item not found.', 'danger')
        return redirect(url_for('dashboard'))
        
    p.is_public = True
    db.session.commit()
    flash('Successfully published your verification to the global Community Feed!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/prediction/delete/<int:pid>', methods=['POST'])
@login_required
def delete_prediction(pid):
    p = Prediction.query.get(pid)
    if not p:
        flash('Item not found', 'danger')
        return redirect(url_for('dashboard'))
    if p.user_id != int(current_user.get_id()):
        flash('Not authorized', 'danger')
        return redirect(url_for('dashboard'))
    try:
        db.session.delete(p)
        db.session.commit()
        flash('Removed from history', 'success')
    except Exception:
        flash('Could not delete item', 'danger')
    return redirect(url_for('dashboard'))


@app.route('/vote/<int:pid>/<action>', methods=['POST'])
@login_required
def vote_prediction(pid, action):
    p = Prediction.query.get(pid)
    if not p:
        return jsonify({'error': 'Prediction not found'}), 404
        
    user_vote = Vote.query.filter_by(user_id=int(current_user.get_id()), prediction_id=pid).first()
    
    if action == 'upvote':
        if user_vote:
            if user_vote.vote_type == 1:
                # UN-VOTE
                db.session.delete(user_vote)
                p.upvotes = max(0, (p.upvotes or 0) - 1)
            else:
                user_vote.vote_type = 1
                p.upvotes = (p.upvotes or 0) + 1
                p.downvotes = max(0, (p.downvotes or 0) - 1)
        else:
            new_vote = Vote(user_id=int(current_user.get_id()), prediction_id=pid, vote_type=1)
            db.session.add(new_vote)
            p.upvotes = (p.upvotes or 0) + 1
            
    elif action == 'downvote':
        if user_vote:
            if user_vote.vote_type == -1:
                # UN-VOTE
                db.session.delete(user_vote)
                p.downvotes = max(0, (p.downvotes or 0) - 1)
            else:
                user_vote.vote_type = -1
                p.downvotes = (p.downvotes or 0) + 1
                p.upvotes = max(0, (p.upvotes or 0) - 1)
        else:
            new_vote = Vote(user_id=int(current_user.get_id()), prediction_id=pid, vote_type=-1)
            db.session.add(new_vote)
            p.downvotes = (p.downvotes or 0) + 1
            
    db.session.commit()
    return jsonify({'upvotes': p.upvotes, 'downvotes': p.downvotes})


@app.route('/analytics')
def analytics():
    # Global analytics
    total_checks = Prediction.query.count()
    fake_checks = Prediction.query.filter_by(prediction='FAKE').count()
    real_checks = Prediction.query.filter_by(prediction='REAL').count()
    
    # Recent community checks with high votes
    trending = Prediction.query.filter((Prediction.upvotes > 0) | (Prediction.downvotes > 0)).order_by(Prediction.upvotes.desc()).limit(10).all()
    
    stats = {
        'total': total_checks,
        'fake': fake_checks,
        'real': real_checks,
        'fake_percent': round((fake_checks/total_checks*100) if total_checks > 0 else 0, 1)
    }
    
    return render_template('analytics.html', stats=stats, trending=trending)


@app.route('/api/ai_news')
def get_ai_news():
    url = "https://news.google.com/rss/search?q=%22Artificial+Intelligence%22+OR+%22AI%22+when:1d&hl=en-US&gl=US&ceid=US:en"
    return _fetch_google_rss(url, limit=5)

@app.route('/api/world_news')
def get_world_news():
    url = "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en"
    return _fetch_google_rss(url, limit=10)

@app.route('/api/related_news', methods=['POST'])
def get_related_news():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    try:
        # We will extract first ~10 words to make a robust search query
        words = query.split()
        search_query = ' '.join(words[:10]) + " news updates"
        results = DDGS().text(search_query, max_results=3)
        out = []
        for r in results:
            out.append({
                'title': r.get('title', ''),
                'href': r.get('href', ''),
                'snippet': r.get('body', '')[:120] + '...'
            })
        return jsonify({'status': 'success', 'articles': out})
    except Exception as e:
        print(f"Related News Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def _fetch_google_rss(url, limit=10):
    try:
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        news_list = []
        for item in items[:limit]:
            title = item.find('title').text if item.find('title') is not None else 'No title'
            link = item.find('link').text if item.find('link') is not None else '#'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            news_list.append({'title': title, 'link': link, 'pubDate': pub_date})
            
        return jsonify({'status': 'success', 'articles': news_list})
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch news'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
