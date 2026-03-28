from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    bio = db.Column(db.Text, nullable=True)
    profile_picture = db.Column(db.String(255), default='https://ui-avatars.com/api/?name=User&background=10b981&color=fff')
    avatar_base64 = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(100), nullable=True)
    joined_date = db.Column(db.DateTime, default=datetime.utcnow)
    theme_preference = db.Column(db.String(10), default='dark')
    city = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    birthdate = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(20), nullable=True)

    def get_id(self):
        return str(self.id)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    
    # Advanced Features columns
    url = db.Column(db.String(1000), nullable=True)
    source_credibility = db.Column(db.Float, nullable=True)
    upvotes = db.Column(db.Integer, default=0)
    downvotes = db.Column(db.Integer, default=0)
    explanation = db.Column(db.Text, nullable=True)
    suspicious_words = db.Column(db.Text, nullable=True) # comma separated list
    
    # Community Feature
    is_public = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref=db.backref('predictions', lazy='dynamic'))

class Vote(db.Model):
    __tablename__ = 'votes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    vote_type = db.Column(db.Integer, nullable=False) # 1 for up, -1 for down
