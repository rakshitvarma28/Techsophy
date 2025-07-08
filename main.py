import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANba2wEAAAAA3Vm%2FKSSluOtfF8jMJufNYoVGxjg%3DxGXzc9lL9tSDxGR6yFMeAUpRSUUQYCQkFgZfRDYwyg0z0mPaoK'
QUERY = "#MentalHealth -is:retweet lang:en"
MAX_RESULTS = 100

#Fetch Tweets
def fetch_tweets():
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    tweets = client.search_recent_tweets(query=QUERY, max_results=MAX_RESULTS,
                                         tweet_fields=["created_at", "text"])
    tweet_data = [(t.created_at, t.text) for t in tweets.data]
    return tweet_data

#Clean Text
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|RT", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

#Sentiment Analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

#Step 4: Build DataFrame
def build_dataframe(tweets):
    data = []
    for timestamp, text in tweets:
        cleaned = clean_text(text)
        score = analyze_sentiment(cleaned)
        data.append({
            "timestamp": timestamp,
            "text": cleaned,
            "sentiment": score
        })
    df = pd.DataFrame(data)
    df.to_csv("sentiment_data.csv", index=False)
    return df

#Plot Sentiment Trend
def plot_trend(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    trend = df['sentiment'].resample('H').mean()

    plt.figure(figsize=(10, 4))
    sns.lineplot(x=trend.index, y=trend.values)
    plt.title("Sentiment Trend on #MentalHealth")
    plt.ylabel("Average Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tweets = fetch_tweets()
    df = build_dataframe(tweets)
    plot_trend(df)
