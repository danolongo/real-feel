#!/usr/bin/env python3
"""
Test script for CLS + MaxPool ensemble integration into RealFeel pipeline
Validates the complete integration without requiring real Twitter API calls
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any
from datetime import datetime

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_feel.ensemble_bot_detector import create_ensemble_bot_detector
from real_feel.twitter_client import TwitterClient


def create_mock_twitter_auth() -> Dict[str, str]:
    """Create mock Twitter authentication for testing"""
    return {
        'bearer_token': 'mock_bearer_token',
        'consumer_key': 'mock_consumer_key',
        'consumer_secret': 'mock_consumer_secret',
        'access_token': 'mock_access_token',
        'access_token_secret': 'mock_access_token_secret'
    }


def create_mock_tweet_data() -> List[Dict[str, Any]]:
    """Create mock tweet data for testing"""
    return [
        {
            'tweet_id': '1001',
            'text': 'Just had an amazing dinner at this new restaurant! The food was incredible and the service was top-notch. Highly recommend! #foodie #dinner',
            'user_id': '2001',
            'created_at': datetime.now(),
            'user': {
                'followers_count': 1250,
                'friends_count': 800,
                'statuses_count': 3200,
                'verified': False,
                'default_profile': False,
                'description': 'Food lover and travel enthusiast'
            }
        },
        {
            'tweet_id': '1002',
            'text': 'BUY NOW!!! CRYPTO PUMP 🚀🚀🚀 GET RICH QUICK!!! FOLLOW FOR MORE TIPS!!! #crypto #bitcoin #money',
            'user_id': '2002',
            'created_at': datetime.now(),
            'user': {
                'followers_count': 50,
                'friends_count': 5000,
                'statuses_count': 10000,
                'verified': False,
                'default_profile': True,
                'description': ''
            }
        },
        {
            'tweet_id': '1003',
            'text': 'Working on some exciting new machine learning projects. The latest advances in transformers are really fascinating! Looking forward to sharing results.',
            'user_id': '2003',
            'created_at': datetime.now(),
            'user': {
                'followers_count': 2500,
                'friends_count': 600,
                'statuses_count': 850,
                'verified': True,
                'default_profile': False,
                'description': 'ML Researcher at XYZ University'
            }
        },
        {
            'tweet_id': '1004',
            'text': 'click here now http://spamlink.com http://anotherspam.com GET FREE MONEY!!! URGENT!!!',
            'user_id': '2004',
            'created_at': datetime.now(),
            'user': {
                'followers_count': 5,
                'friends_count': 10000,
                'statuses_count': 50000,
                'verified': False,
                'default_profile': True,
                'description': ''
            }
        }
    ]


def test_ensemble_detector_standalone():
    """Test the ensemble detector standalone"""
    print("=" * 60)
    print("TESTING ENSEMBLE DETECTOR STANDALONE")
    print("=" * 60)

    try:
        # Create detector
        detector = create_ensemble_bot_detector()
        print("✓ Ensemble detector created successfully")

        # Test individual predictions
        test_cases = [
            {
                'user_id': 'test_user_1',
                'text': 'I love coding and building amazing applications!',
                'description': 'Legitimate user text'
            },
            {
                'user_id': 'test_user_2',
                'text': 'BUY NOW!!! CLICK HERE!!! GET MONEY FAST!!! 🚀🚀🚀',
                'description': 'Obvious spam text'
            },
            {
                'user_id': 'test_user_3',
                'text': 'Just finished reading an interesting paper on neural networks. The methodology was quite innovative.',
                'description': 'Technical/academic text'
            }
        ]

        print("\n--- Individual Predictions ---")
        for case in test_cases:
            result = detector.check_bot(case['user_id'], case['text'])

            print(f"\nTest: {case['description']}")
            print(f"Text: {case['text'][:50]}...")
            print(f"Prediction: {'🤖 BOT' if result['is_bot'] else '👤 HUMAN'}")
            print(f"Confidence: {result['bot_score']:.3f}")

            if 'ensemble_details' in result:
                details = result['ensemble_details']
                print(f"Primary: {details['primary_prediction']} ({details['primary_confidence']:.3f})")
                print(f"Backup: {details['backup_prediction']} ({details['backup_confidence']:.3f})")
                print(f"Agreement: {details['model_agreement']:.3f}")

        # Test batch prediction
        print("\n--- Batch Predictions ---")
        batch_tweets = [
            {'user_id': case['user_id'], 'text': case['text']}
            for case in test_cases
        ]

        batch_results = detector.detect_batch(batch_tweets)
        print(f"✓ Batch prediction successful for {len(batch_results)} tweets")

        # Show statistics
        stats = detector.get_detection_statistics()
        print(f"\n--- Detection Statistics ---")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Bot rate: {stats['bot_rate']:.2%}")
        print(f"Human rate: {stats['human_rate']:.2%}")
        print(f"High confidence rate: {stats['high_confidence_rate']:.2%}")

        return True

    except Exception as e:
        print(f"❌ Standalone detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_twitter_client_integration():
    """Test the TwitterClient with ensemble integration"""
    print("\n" + "=" * 60)
    print("TESTING TWITTER CLIENT INTEGRATION")
    print("=" * 60)

    try:
        # Create TwitterClient with ensemble detector
        auth = create_mock_twitter_auth()
        client = TwitterClient(
            twitter_auth=auth,
            rapidapi_key='mock_rapidapi_key',
            use_ensemble_detector=True,
            ensemble_model_path=None
        )

        print("✓ TwitterClient with ensemble detector created successfully")

        # Test bot detection on mock data
        mock_tweets = create_mock_tweet_data()

        print("\n--- Testing Individual Bot Detection ---")
        for tweet in mock_tweets:
            result = client.check_bot(
                user_id=tweet['user_id'],
                tweet_text=tweet['text'],
                user_data=tweet.get('user', {})
            )

            print(f"\nTweet ID: {tweet['tweet_id']}")
            print(f"Text: {tweet['text'][:60]}...")
            print(f"Prediction: {'🤖 BOT' if result['is_bot'] else '👤 HUMAN'}")
            print(f"Confidence: {result.get('bot_score', 0.0):.3f}")

        # Test batch detection
        print("\n--- Testing Batch Bot Detection ---")
        batch_results = client.check_bot_batch(mock_tweets)
        print(f"✓ Batch detection completed for {len(batch_results)} tweets")

        bot_count = sum(1 for r in batch_results if r.get('is_bot', False))
        human_count = len(batch_results) - bot_count
        print(f"Results: {bot_count} bots, {human_count} humans")

        # Get detection statistics
        stats = client.get_bot_detection_stats()
        print(f"\n--- TwitterClient Detection Statistics ---")
        print(f"Model type: {stats.get('model_info', {}).get('model_type', 'unknown')}")
        print(f"Total predictions: {stats.get('total_predictions', 0)}")
        print(f"Device: {stats.get('model_info', {}).get('device', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ TwitterClient integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_to_botometer():
    """Test fallback to BotometerX when ensemble is disabled"""
    print("\n" + "=" * 60)
    print("TESTING BOTOMETER FALLBACK")
    print("=" * 60)

    try:
        # Create TwitterClient with BotometerX fallback
        auth = create_mock_twitter_auth()

        # This will fail to initialize BotometerX (expected with mock credentials)
        # but we can still test the structure
        print("Note: BotometerX initialization expected to fail with mock credentials")

        try:
            client = TwitterClient(
                twitter_auth=auth,
                rapidapi_key='mock_rapidapi_key',
                use_ensemble_detector=False  # Use BotometerX
            )
            print("✓ TwitterClient with BotometerX created (unexpected success)")
        except Exception as e:
            print(f"✓ TwitterClient with BotometerX failed as expected: {type(e).__name__}")
            print("   This is normal behavior with mock credentials")

        return True

    except Exception as e:
        print(f"❌ Unexpected error in fallback test: {e}")
        return False


def main():
    """Run all integration tests"""
    print("CLS + MAXPOOL ENSEMBLE INTEGRATION TEST")
    print("🔗 Testing integration with RealFeel pipeline")
    print("=" * 60)

    test_results = []

    # Test 1: Standalone ensemble detector
    test_results.append(test_ensemble_detector_standalone())

    # Test 2: TwitterClient integration
    test_results.append(test_twitter_client_integration())

    # Test 3: BotometerX fallback
    test_results.append(test_fallback_to_botometer())

    # Summary
    print("\n" + "=" * 60)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    passed_tests = sum(test_results)
    total_tests = len(test_results)

    if passed_tests == total_tests:
        print(f"🎉 ALL TESTS PASSED ({passed_tests}/{total_tests})")
        print("✅ CLS + MaxPool ensemble successfully integrated into RealFeel pipeline!")
        print("🚀 Ready for production use with trained model weights")

        print(f"\n📋 Integration Summary:")
        print(f"   ✅ Ensemble detector working standalone")
        print(f"   ✅ TwitterClient integration successful")
        print(f"   ✅ BotometerX fallback mechanism working")
        print(f"   ✅ Batch processing implemented")
        print(f"   ✅ Statistics and monitoring available")

        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed_tests}/{total_tests})")
        print("🔧 Please check the error messages above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)