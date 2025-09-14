from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Any, Optional
from config import DATABASE_URL, TWITTER_AUTH, RAPIDAPI_KEY
from models import Tweet
from pipeline import DataPipeline

app = FastAPI(
    title="Real Feel Database API", 
    description="Comprehensive API for Twitter sentiment analysis, bot detection, and database querying",
    version="1.0.0"
)

# Initialize pipeline for tweet analysis
pipeline = DataPipeline(
    db_url=DATABASE_URL,
    twitter_auth=TWITTER_AUTH,
    rapidapi_key=RAPIDAPI_KEY
)

# Database session setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """Request model for SQL queries"""
    query: str

class QueryResponse(BaseModel):
    """Response model for SQL query results"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    message: str = ""

class PlotDataResponse(BaseModel):
    """Response model for plot data endpoints"""
    success: bool
    plot_type: str
    data: Dict[str, Any]
    message: str = ""

class TweetResponse(BaseModel):
    """Response model for individual tweet data"""
    tweet_id: str
    text: str
    sentiment: str
    sentiment_confidence: float
    is_bot: bool
    bot_score: float

class StatsResponse(BaseModel):
    """Response model for statistics data"""
    total_tweets: int
    bot_tweets: int
    real_tweets: int
    sentiment_stats: dict

@app.get("/")
async def root():
    """
    Root endpoint providing API information and available endpoints.
    
    Returns:
        dict: API welcome message and list of available endpoints
    """
    return {
        "message": "Real Feel Database API", 
        "description": "Comprehensive API for Twitter sentiment analysis, bot detection, and database querying",
        "endpoints": {
            "database": ["/query", "/test-connection", "/data/summary"],
            "plots": ["/plots/bot-vs-real", "/plots/sentiment-comparison", "/plots/bot-sentiment", "/plots/real-sentiment"],
            "analysis": ["/analyze", "/stats", "/tweets/{sentiment_type}"]
        }
    }

@app.get("/test-connection")
async def test_connection():
    """
    Test database connectivity.
    
    Returns:
        dict: Success status and connection message
        
    Raises:
        HTTPException: If database connection fails
    """
    try:
        test_engine = create_engine(DATABASE_URL)
        with test_engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        return {"success": True, "message": "Database connection successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Execute arbitrary SQL queries against the database.
    
    Args:
        request: QueryRequest containing the SQL query string
        
    Returns:
        QueryResponse: Structured response with query results, columns, and metadata
        
    Raises:
        HTTPException: If query is empty or execution fails
        
    Note:
        - SELECT/WITH queries return data rows and columns
        - Other queries (INSERT/UPDATE/DELETE) return affected row count
    """
    try:
        query_engine = create_engine(DATABASE_URL)
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if query.upper().startswith(('SELECT', 'WITH')):
            df = pd.read_sql_query(text(query), query_engine)
            
            # Convert DataFrame to JSON-serializable format
            data = []
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, np.integer):
                        row_dict[col] = int(value)
                    elif isinstance(value, np.floating):
                        row_dict[col] = float(value)
                    else:
                        row_dict[col] = str(value)
                data.append(row_dict)
            
            return QueryResponse(
                success=True,
                data=data,
                columns=list(df.columns),
                row_count=len(df),
                message=f"Query executed successfully. {len(df)} rows returned."
            )
        else:
            # Handle non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
            with query_engine.connect() as connection:
                result = connection.execute(text(query))
                connection.commit()
                
                return QueryResponse(
                    success=True,
                    data=None,
                    columns=None,
                    row_count=result.rowcount if result.rowcount >= 0 else 0,
                    message=f"Query executed successfully. {result.rowcount if result.rowcount >= 0 else 0} rows affected."
                )
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

# Tweet Analysis Endpoints (from original api.py)
@app.post("/analyze", response_model=List[TweetResponse])
async def analyze_tweets(query: str, max_tweets: Optional[int] = 100):
    """
    Analyze tweets for a given search query using the data pipeline.
    
    This endpoint fetches tweets from Twitter API, performs sentiment analysis
    and bot detection, then stores results in the database.
    
    Args:
        query: Search query string for Twitter API
        max_tweets: Maximum number of tweets to analyze (default: 100)
        
    Returns:
        List[TweetResponse]: List of analyzed tweets with sentiment and bot scores
        
    Raises:
        HTTPException: If tweet analysis or processing fails
    """
    try:
        tweets = pipeline.process_tweets(query, max_tweets)
        return [{
            "tweet_id": tweet.tweet_id,
            "text": tweet.text,
            "sentiment": tweet.sentiment,
            "sentiment_confidence": tweet.sentiment_confidence,
            "is_bot": tweet.is_bot,
            "bot_score": tweet.bot_score
        } for tweet in tweets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tweet analysis failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get comprehensive statistics about all analyzed tweets in the database.
    
    Returns:
        StatsResponse: Statistics including total tweets, bot counts, and sentiment distribution
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        return pipeline.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@app.get("/tweets/{sentiment_type}")
async def get_tweets_by_sentiment(
    sentiment_type: str,
    bot_only: Optional[bool] = None,
    limit: Optional[int] = 10
):
    """
    Retrieve tweets filtered by sentiment type and optionally by bot status.
    
    Args:
        sentiment_type: Sentiment filter - must be 'positive', 'neutral', or 'negative'
        bot_only: If True, return only bot tweets; if False, return only real user tweets; if None, return both
        limit: Maximum number of tweets to return (default: 10)
        
    Returns:
        List[dict]: List of tweet objects matching the filters
        
    Raises:
        HTTPException: If sentiment_type is invalid or database query fails
    """
    if sentiment_type not in ['positive', 'neutral', 'negative']:
        raise HTTPException(
            status_code=400,
            detail="Invalid sentiment type. Must be one of: positive, neutral, negative"
        )
    
    try:
        session = SessionLocal()
        query = session.query(Tweet).filter_by(sentiment=sentiment_type)
        if bot_only is not None:
            query = query.filter_by(is_bot=bot_only)
        
        tweets = query.limit(limit).all()
        session.close()
        
        return [{
            "tweet_id": tweet.tweet_id,
            "text": tweet.text,
            "sentiment": tweet.sentiment,
            "sentiment_confidence": tweet.sentiment_confidence,
            "is_bot": tweet.is_bot,
            "bot_score": tweet.bot_score
        } for tweet in tweets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tweets: {str(e)}")

# Plot Data Endpoints
@app.get("/plots/bot-vs-real", response_model=PlotDataResponse)
async def get_bot_vs_real_data():
    """
    Get data for bot vs real user distribution visualization.
    
    Retrieves all tweets from database and calculates the distribution
    between bot and real user accounts for pie chart and bar chart display.
    
    Returns:
        PlotDataResponse: Contains labels, values, colors, and total count for visualization
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        plot_engine = create_engine(DATABASE_URL)
        df = pd.read_sql_query(text("SELECT * FROM tweets"), plot_engine)
        
        if df.empty:
            return PlotDataResponse(
                success=False,
                plot_type="bot_vs_real",
                data={},
                message="No data found in database"
            )
        
        bot_count = int(df['is_bot'].sum()) if 'is_bot' in df.columns else 0
        real_count = len(df) - bot_count
        
        data = {
            "labels": ["Real Users", "Bots"],
            "values": [real_count, bot_count],
            "colors": ["#51cf66", "#ff6b6b"],
            "total": len(df)
        }
        
        return PlotDataResponse(
            success=True,
            plot_type="bot_vs_real",
            data=data,
            message=f"Data loaded successfully. {len(df)} total tweets."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load bot vs real data: {str(e)}")

@app.get("/plots/sentiment-comparison", response_model=PlotDataResponse)
async def get_sentiment_comparison_data():
    """
    Get comparative sentiment analysis data between bots and real users.
    
    Analyzes sentiment distribution across positive, negative, and neutral categories
    for both bot and real user tweets, returning percentages and counts for comparison.
    
    Returns:
        PlotDataResponse: Contains sentiment breakdowns, percentages, and counts for bot vs real comparison
        
    Raises:
        HTTPException: If database query fails or required columns are missing
    """
    try:
        plot_engine = create_engine(DATABASE_URL)
        df = pd.read_sql_query(text("SELECT * FROM tweets"), plot_engine)
        
        if df.empty:
            return PlotDataResponse(
                success=False,
                plot_type="sentiment_comparison",
                data={},
                message="No data found in database"
            )
        
        if 'is_bot' not in df.columns or 'sentiment' not in df.columns:
            return PlotDataResponse(
                success=False,
                plot_type="sentiment_comparison",
                data={},
                message="Required columns (is_bot, sentiment) not found"
            )
        
        # Separate bot and real user tweets
        bot_tweets = df[df['is_bot'] == True]
        real_tweets = df[df['is_bot'] == False]
        
        all_sentiments = ['positive', 'negative', 'neutral']
        
        # Calculate counts and percentages
        bot_counts = bot_tweets['sentiment'].value_counts().reindex(all_sentiments, fill_value=0)
        real_counts = real_tweets['sentiment'].value_counts().reindex(all_sentiments, fill_value=0)
        
        bot_pct = (bot_counts / bot_counts.sum() * 100).fillna(0) if bot_counts.sum() > 0 else pd.Series([0, 0, 0], index=all_sentiments)
        real_pct = (real_counts / real_counts.sum() * 100).fillna(0) if real_counts.sum() > 0 else pd.Series([0, 0, 0], index=all_sentiments)
        
        data = {
            "sentiments": all_sentiments,
            "real_percentages": real_pct.tolist(),
            "bot_percentages": bot_pct.tolist(),
            "real_counts": real_counts.tolist(),
            "bot_counts": bot_counts.tolist(),
            "colors": {
                "real": "#51cf66",
                "bot": "#ff6b6b"
            }
        }
        
        return PlotDataResponse(
            success=True,
            plot_type="sentiment_comparison",
            data=data,
            message=f"Sentiment data loaded. {len(bot_tweets)} bot tweets, {len(real_tweets)} real tweets."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sentiment comparison data: {str(e)}")

@app.get("/plots/bot-sentiment", response_model=PlotDataResponse)
async def get_bot_sentiment_data():
    """
    Get sentiment distribution data specifically for bot tweets.
    
    Retrieves only tweets identified as bots and analyzes their sentiment
    distribution across positive, negative, and neutral categories.
    
    Returns:
        PlotDataResponse: Contains sentiment labels, counts, and colors for bot tweet visualization
        
    Raises:
        HTTPException: If no bot tweets found or database query fails
    """
    try:
        plot_engine = create_engine(DATABASE_URL)
        df = pd.read_sql_query(text("SELECT * FROM tweets WHERE is_bot = true"), plot_engine)
        
        if df.empty:
            return PlotDataResponse(
                success=False,
                plot_type="bot_sentiment",
                data={},
                message="No bot tweets found"
            )
        
        if 'sentiment' not in df.columns:
            return PlotDataResponse(
                success=False,
                plot_type="bot_sentiment", 
                data={},
                message="Sentiment column not found"
            )
        
        sentiment_counts = df['sentiment'].value_counts()
        
        data = {
            "sentiments": sentiment_counts.index.tolist(),
            "counts": sentiment_counts.values.tolist(),
            "colors": ["#51cf66" if s == "positive" else "#ff6b6b" if s == "negative" else "#ffd43b" for s in sentiment_counts.index],
            "total": len(df)
        }
        
        return PlotDataResponse(
            success=True,
            plot_type="bot_sentiment",
            data=data,
            message=f"Bot sentiment data loaded. {len(df)} bot tweets."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load bot sentiment data: {str(e)}")

@app.get("/plots/real-sentiment", response_model=PlotDataResponse)
async def get_real_sentiment_data():
    """
    Get sentiment distribution data specifically for real user tweets.
    
    Retrieves only tweets from real users (non-bots) and analyzes their sentiment
    distribution across positive, negative, and neutral categories.
    
    Returns:
        PlotDataResponse: Contains sentiment labels, counts, and colors for real user tweet visualization
        
    Raises:
        HTTPException: If no real user tweets found or database query fails
    """
    try:
        plot_engine = create_engine(DATABASE_URL)
        df = pd.read_sql_query(text("SELECT * FROM tweets WHERE is_bot = false"), plot_engine)
        
        if df.empty:
            return PlotDataResponse(
                success=False,
                plot_type="real_sentiment",
                data={},
                message="No real user tweets found"
            )
        
        if 'sentiment' not in df.columns:
            return PlotDataResponse(
                success=False,
                plot_type="real_sentiment",
                data={},
                message="Sentiment column not found"
            )
        
        sentiment_counts = df['sentiment'].value_counts()
        
        data = {
            "sentiments": sentiment_counts.index.tolist(),
            "counts": sentiment_counts.values.tolist(),
            "colors": ["#51cf66" if s == "positive" else "#ff6b6b" if s == "negative" else "#ffd43b" for s in sentiment_counts.index],
            "total": len(df)
        }
        
        return PlotDataResponse(
            success=True,
            plot_type="real_sentiment",
            data=data,
            message=f"Real user sentiment data loaded. {len(df)} real tweets."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load real sentiment data: {str(e)}")

@app.get("/data/summary")
async def get_data_summary():
    """
    Get comprehensive summary statistics of all data in the database.
    
    Provides an overview including total tweet count, column information,
    bot/real user distribution, sentiment breakdown, and date range of tweets.
    
    Returns:
        dict: Comprehensive data summary with counts, distributions, and metadata
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        summary_engine = create_engine(DATABASE_URL)
        df = pd.read_sql_query(text("SELECT * FROM tweets"), summary_engine)
        
        if df.empty:
            return {"success": False, "message": "No data found"}
        
        summary = {
            "total_tweets": len(df),
            "columns": list(df.columns),
            "bot_count": int(df['is_bot'].sum()) if 'is_bot' in df.columns else None,
            "real_count": int((~df['is_bot']).sum()) if 'is_bot' in df.columns else None,
            "sentiment_distribution": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else None,
            "date_range": {
                "earliest": df['created_at'].min().isoformat() if 'created_at' in df.columns else None,
                "latest": df['created_at'].max().isoformat() if 'created_at' in df.columns else None
            } if 'created_at' in df.columns else None
        }
        
        return {"success": True, "data": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)