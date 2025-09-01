import sys
import pandas as pd
from pathlib import Path

def setup_path():
    """Setup Python path to find our modules"""
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / "src"
    
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"Added {src_path} to Python path")
    
    return project_root, src_path

def create_sample_data():
    """Create sample data for testing plotting functions"""
    
    # Sample tweet data
    sample_tweets = [
        {
            'id': '1',
            'text': 'I love this product! Amazing quality!',
            'user': {
                'screen_name': 'real_user1',
                'followers_count': 150,
                'friends_count': 200,
                'statuses_count': 1000,
                'verified': False,
                'default_profile': False,
                'default_profile_image': False,
                'description': 'Real person who loves tech'
            }
        },
        {
            'id': '2', 
            'text': 'BUY NOW!!! AMAZING CRYPTO OPPORTUNITY!!! 🚀🚀🚀',
            'user': {
                'screen_name': 'cryptobot123',
                'followers_count': 5,
                'friends_count': 5000,
                'statuses_count': 10000,
                'verified': False,
                'default_profile': True,
                'default_profile_image': True,
                'description': ''
            }
        },
        {
            'id': '3',
            'text': 'This weather is terrible today',
            'user': {
                'screen_name': 'weather_person',
                'followers_count': 300,
                'friends_count': 250,
                'statuses_count': 800,
                'verified': False,
                'default_profile': False,
                'default_profile_image': False,
                'description': 'Weather enthusiast'
            }
        },
        {
            'id': '4',
            'text': 'PUMP PUMP PUMP!!! Buy $SCAMCOIN now!!!',
            'user': {
                'screen_name': 'pump_bot_456',
                'followers_count': 2,
                'friends_count': 8000,
                'statuses_count': 20000,
                'verified': False,
                'default_profile': True,
                'default_profile_image': True,
                'description': ''
            }
        },
        {
            'id': '5',
            'text': 'Having a great day with friends!',
            'user': {
                'screen_name': 'happy_human',
                'followers_count': 400,
                'friends_count': 350,
                'statuses_count': 1200,
                'verified': False,
                'default_profile': False,
                'default_profile_image': False,
                'description': 'Just living life'
            }
        }
    ]
    
    return sample_tweets

def main():
    """Main test function with plotting"""
    print("=== Real Feel Sentiment Analysis Test with Plotting ===\n")
    
    # Setup paths
    project_root, src_path = setup_path()
    
    # Test imports
    try:
        from src.real_feel.sentimentPredict import SentimentAnalyzer, BotDetector
        from src.real_feel.dataProcessing import DataProcessor
        from src.real_feel.plotting import SentimentVisualizer
        print("Successfully imported modules")
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    print("\n=== Initializing Components ===")
    try:
        # Initialize components
        print("Loading sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer()
        
        print("Loading bot detector...")
        bot_detector = BotDetector()
        
        print("Loading data processor...")
        data_processor = DataProcessor()
        
        print("Loading sentiment visualizer...")
        visualizer = SentimentVisualizer()
        
        print("All components loaded successfully")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    # Create sample data
    print("\n=== Creating Sample Data ===")
    sample_tweets = create_sample_data()
    print(f"Created {len(sample_tweets)} sample tweets")
    
    # Analyze all sample tweets
    print("\n=== Analyzing Sample Tweets ===")
    bot_predictions = []
    sentiment_predictions = []
    
    for i, tweet in enumerate(sample_tweets):
        print(f"Processing tweet {i+1}/{len(sample_tweets)}: '{tweet['text'][:50]}...'")
        
        # Bot detection
        bot_result = bot_detector.isItBot(tweet)
        bot_predictions.append(bot_result)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.sentimentAnalysis(tweet['text'])
        sentiment_predictions.append(sentiment_result)
        
        print(f"  Bot: {bot_result['is_bot']} (confidence: {bot_result['bot_confidence']:.3f})")
        print(f"  Sentiment: {sentiment_result['predicted_sentiment']} (confidence: {sentiment_result['confidence']:.3f})")
    
    # Test amounts function
    print(f"\n=== Testing amounts() function ===")
    bot_count, real_count, total_count = data_processor.amounts(bot_predictions)
    print(f"Bot tweets: {bot_count}")
    print(f"Real tweets: {real_count}")
    print(f"Total tweets: {total_count}")
    
    # Test plotting functions
    print(f"\n=== Testing Plotting Functions ===")
    
    # Create plots directory
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # Test botVSreal()
        print("Creating botVSreal() plot...")
        visualizer.botVSreal(bot_predictions, save_path=str(plots_dir / "bot_vs_real.png"))
        
        # Test botSentiment()
        print("Creating botSentiment() plot...")
        visualizer.botSentiment(bot_predictions, sentiment_predictions, 
                                save_path=str(plots_dir / "bot_sentiment.png"))
        
        # Test realSentiment()
        print("Creating realSentiment() plot...")
        visualizer.realSentiment(bot_predictions, sentiment_predictions,
                                save_path=str(plots_dir / "real_sentiment.png"))
        
        # Test comparison plot
        print("Creating comparison plot...")
        visualizer.compare_all_sentiments(bot_predictions, sentiment_predictions,
                                        save_path=str(plots_dir / "sentiment_comparison.png"))
        
        print("All plots created successfully")
        print(f"Plots saved in: {plots_dir}")
        
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n All tests completed successfully")
    print("Analysis Summary:")
    print(f"   - Total tweets analyzed: {total_count}")
    print(f"   - Bot tweets detected: {bot_count}")
    print(f"   - Real user tweets: {real_count}")
    
    # Show sentiment breakdown
    sentiments = [pred['predicted_sentiment'] for pred in sentiment_predictions]
    sentiment_counts = pd.Series(sentiments).value_counts()
    print(f"   - Sentiment breakdown:")
    for sentiment, count in sentiment_counts.items():
        print(f"     • {sentiment}: {count}")
    
    print("\n Your Real Feel sentiment analysis and plotting system is working")
    print(f"Check the plots in the '{plots_dir}' directory")

if __name__ == "__main__":
    main()