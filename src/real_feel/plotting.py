import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class SentimentVisualizer:
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        """        
        Args:
            style (str): Matplotlib style to use
            figsize (tuple): Default figure size for plots
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            
        self.figsize = figsize
        self.colors = {
            'bot': '#ff6b6b',      # Red for bots
            'real': '#51cf66',     # Green for real users  
            'positive': '#51cf66', # Green for positive
            'negative': '#ff6b6b', # Red for negative
            'neutral': '#ffd43b'   # Yellow for neutral
        }
    
    def botVSreal(self, predictions, save_path=None):
        """
        Create visualization comparing bot vs real tweet distributions
        
        Args:
            predictions (list): List of bot detection predictions with 'is_bot' key
            save_path (str, optional): Path to save the plot
        """
        # Extract bot/real counts
        bot_count = sum(1 for pred in predictions if pred.get('is_bot', False))
        real_count = len(predictions) - bot_count
        
        if bot_count == 0 and real_count == 0:
            print("No data to plot in botVSreal")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Data for plotting
        labels = ['Real Users', 'Bots']
        sizes = [real_count, bot_count]
        colors = [self.colors['real'], self.colors['bot']]
        
        # Pie chart
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                                autopct='%1.1f%%', startangle=90)
            ax1.set_title('Bot vs Real Users Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8)
        ax2.set_title('Bot vs Real Users Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Tweets')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                    f'{int(size)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def botSentiment(self, bot_predictions, sentiment_predictions, save_path=None):
        """
        Visualize sentiment distribution for bot tweets only
        
        Args:
            bot_predictions (list): Bot detection results with 'is_bot' key
            sentiment_predictions (list): Sentiment analysis results with 'predicted_sentiment' key
            save_path (str, optional): Path to save the plot
        """
        # Filter for bot tweets only
        bot_sentiments = []
        for bot_pred, sent_pred in zip(bot_predictions, sentiment_predictions):
            if bot_pred.get('is_bot', False):
                sentiment = sent_pred.get('predicted_sentiment', 'neutral')
                bot_sentiments.append(sentiment)
        
        if not bot_sentiments:
            print("No bot tweets found for sentiment analysis")
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Bot Tweets Found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            ax.set_title('Bot Tweets Sentiment Distribution')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            return
        
        self._plot_sentiment_distribution(bot_sentiments, "Bot Tweets Sentiment Distribution", save_path)
    
    def realSentiment(self, bot_predictions, sentiment_predictions, save_path=None):
        """
        Visualize sentiment distribution for real user tweets only
        
        Args:
            bot_predictions (list): Bot detection results with 'is_bot' key
            sentiment_predictions (list): Sentiment analysis results with 'predicted_sentiment' key
            save_path (str, optional): Path to save the plot
        """
        # Filter for real user tweets only
        real_sentiments = []
        for bot_pred, sent_pred in zip(bot_predictions, sentiment_predictions):
            if not bot_pred.get('is_bot', False):
                sentiment = sent_pred.get('predicted_sentiment', 'neutral')
                real_sentiments.append(sentiment)
        
        if not real_sentiments:
            print("No real user tweets found for sentiment analysis")
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Real User Tweets Found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            ax.set_title('Real Users Sentiment Distribution')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            return
        
        self._plot_sentiment_distribution(real_sentiments, "Real Users Sentiment Distribution", save_path)
    
    def _plot_sentiment_distribution(self, sentiments, title, save_path=None):
        """
        Helper method to plot sentiment distribution
        
        Args:
            sentiments (list): List of sentiment labels
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        if not sentiments:
            print(f"No sentiment data to plot for {title}")
            return
        
        # Count sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Bar chart
        colors_list = [self.colors.get(sent, '#808080') for sent in sentiment_counts.index]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                        color=colors_list, alpha=0.8)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Tweets')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(sentiment_counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(sentiment_counts.values, 
                                            labels=sentiment_counts.index,
                                            colors=colors_list,
                                            autopct='%1.1f%%', 
                                            startangle=90)
        ax2.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def compare_all_sentiments(self, bot_predictions, sentiment_predictions, save_path=None):
        """
        Create a comprehensive comparison of bot vs real sentiment distributions
        
        Args:
            bot_predictions (list): Bot detection results
            sentiment_predictions (list): Sentiment analysis results
            save_path (str, optional): Path to save the plot
        """
        # Separate bot and real sentiments
        bot_sentiments = []
        real_sentiments = []
        
        for bot_pred, sent_pred in zip(bot_predictions, sentiment_predictions):
            sentiment = sent_pred.get('predicted_sentiment', 'neutral')
            if bot_pred.get('is_bot', False):
                bot_sentiments.append(sentiment)
            else:
                real_sentiments.append(sentiment)
        
        # Create comparison dataframe
        all_sentiments = ['positive', 'negative', 'neutral']
        
        bot_counts = pd.Series(bot_sentiments).value_counts().reindex(all_sentiments, fill_value=0)
        real_counts = pd.Series(real_sentiments).value_counts().reindex(all_sentiments, fill_value=0)
        
        # Normalize to percentages
        bot_pct = (bot_counts / bot_counts.sum() * 100) if bot_counts.sum() > 0 else bot_counts
        real_pct = (real_counts / real_counts.sum() * 100) if real_counts.sum() > 0 else real_counts
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(all_sentiments))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_pct, width, label='Real Users', 
                        color=self.colors['real'], alpha=0.8)
        bars2 = ax.bar(x + width/2, bot_pct, width, label='Bots', 
                        color=self.colors['bot'], alpha=0.8)
        
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Percentage of Tweets')
        ax.set_title('Sentiment Distribution: Bots vs Real Users')
        ax.set_xticks(x)
        ax.set_xticklabels(all_sentiments)
        ax.legend()
        
        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add label if there's a bar
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
