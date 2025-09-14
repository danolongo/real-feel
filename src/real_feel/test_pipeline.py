from real_feel.pipeline import DataPipeline
from real_feel.config import DATABASE_URL, TWITTER_AUTH, RAPIDAPI_KEY
import json


def process_tweets_with_analysis(pipeline, query, max_tweets=10):
    """
    Process tweets and return analysis results.
    
    Args:
        pipeline: DataPipeline instance for processing tweets
        query: Search query string
        max_tweets: Maximum number of tweets to process
        
    Returns:
        List of processed tweet objects with analysis data
    """
    tweets = pipeline.process_tweets(query, max_tweets=max_tweets)
    return tweets


def get_pipeline_statistics(pipeline):
    """
    Get and format pipeline statistics.
    
    Args:
        pipeline: DataPipeline instance to get statistics from
        
    Returns:
        Dictionary containing pipeline statistics
    """
    stats = pipeline.get_statistics()
    return stats


def display_processed_tweets(tweets, amount=3):
    """
    Display processed tweets with their analysis.
    
    Args:
        tweets: List of processed tweet objects
        amount: Number of tweets to display (default: 3)
    """
    if not tweets:
        print("\nNo tweets to display")
        return
        
    print(f"\nExample processed tweets (showing {min(amount, len(tweets))} of {len(tweets)}):")
    for i, tweet in enumerate(tweets[:amount], 1):
        print(f"\n--- Tweet {i} ---")
        print(f"ID: {tweet.tweet_id}")
        print(f"Text: {tweet.text[:100] + '...' if len(tweet.text) > 100 else tweet.text}")
        print(f"Sentiment: {tweet.sentiment}")
        print(f"Bot Score: {tweet.bot_score:.3f}")
        print(f"Is Bot: {tweet.is_bot}")


def test_pipeline():
    """
    Main test pipeline function that orchestrates tweet processing and analysis.
    
    Prompts user for input, processes tweets, displays statistics and sample results.
    """
    try:
        # Initialize pipeline
        pipeline = DataPipeline(
            db_url=DATABASE_URL,
            twitter_auth=TWITTER_AUTH,
            rapidapi_key=RAPIDAPI_KEY
        )
        
        # Get user input
        query = input("Enter search query: ")
        max_tweets = int(input("Enter max tweets (default 10): ") or "10")
        
        # Process tweets
        print(f"\nProcessing tweets for query: '{query}'...")
        tweets = process_tweets_with_analysis(pipeline, query, max_tweets)
        
        if tweets:
            print(f"✓ Successfully processed {len(tweets)} tweets")
            
            # Get statistics
            print("\nRetrieving pipeline statistics...")
            stats = get_pipeline_statistics(pipeline)
            
            print("\n" + "="*50)
            print("PIPELINE STATISTICS")
            print("="*50)
            print(json.dumps(stats, indent=2))
            
            # Display sample tweets
            display_processed_tweets(tweets, amount=3)
            
        else:
            print("⚠ No tweets were processed")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\n✗ Error in pipeline test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pipeline()
