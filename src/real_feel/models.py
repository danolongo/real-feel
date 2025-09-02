from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Tweet(Base):
    __tablename__ = 'tweets'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, unique=True)
    text = Column(String)
    user_id = Column(String)
    created_at = Column(DateTime)
    
    # Sentiment Analysis fields
    sentiment = Column(String)
    sentiment_confidence = Column(Float)
    sentiment_scores = Column(String)  # JSON string of scores
    
    # Bot Detection fields
    is_bot = Column(Boolean)
    bot_score = Column(Float)
    bot_scores = Column(String)  # JSON string of detailed scores
    
    def __repr__(self):
        return f"<Tweet(id={self.tweet_id}, sentiment={self.sentiment}, is_bot={self.is_bot})>"

def init_db(db_url):
    """Initialize database connection and create tables"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
