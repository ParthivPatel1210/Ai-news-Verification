from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

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

    user = db.relationship('User', backref=db.backref('predictions', lazy='dynamic'))
