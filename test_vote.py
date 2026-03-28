from app import app, db
from models import User, Prediction, Vote
import json

with app.app_context():
    # Let's see if there are any Predictions
    p = Prediction.query.first()
    if p:
        print(f"Prediction {p.id}: {p.upvotes} up, {p.downvotes} down")
        
        # Are there any votes?
        votes = Vote.query.filter_by(prediction_id=p.id).all()
        print(f"Votes for p={p.id}: {[(v.id, v.user_id, v.vote_type) for v in votes]}")
        
        # Test the query
        user = User.query.first()
        if user:
            print(f"Testing vote for user {user.id} on prediction {p.id}")
            # Try upvoting programmatically
            user_vote = Vote.query.filter_by(user_id=user.id, prediction_id=p.id).first()
            if not user_vote:
                new_vote = Vote(user_id=user.id, prediction_id=p.id, vote_type=1)
                db.session.add(new_vote)
                p.upvotes = (p.upvotes or 0) + 1
                db.session.commit()
                print(f"After upvote: {p.upvotes} up, {p.downvotes} down")
            else:
                user_vote.vote_type = -1
                p.downvotes = (p.downvotes or 0) + 1
                p.upvotes = max(0, (p.upvotes or 0) - 1)
                db.session.commit()
                print(f"After toggling downvote: {p.upvotes} up, {p.downvotes} down")
    else:
        print("No predictions found in DB")
