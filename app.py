import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Social Media Sentiment Dashboard - Healthcare")

# Load sentiment data
df = pd.read_csv("sentiment_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Resample by hour
trend = df['sentiment'].resample('h').mean()

# Line Chart
st.subheader("Sentiment Trend Over Time")
st.line_chart(trend)

# Show sample tweets
st.subheader("Sample Tweets with Sentiment Scores")
st.dataframe(df[['text', 'sentiment']].head(10))

st.subheader("Peak Tweet Hours")

# Count number of tweets per hour
tweet_vol = df['text'].resample('h').count()

# Get top 3 highest tweet volume hours
top_peaks = tweet_vol.sort_values(ascending=False).head(3)

for time, count in top_peaks.items():
    st.write(f"{time.strftime('%Y-%m-%d %H:%M')} — {count} tweets")


# Add summary box here
st.markdown("### Summary Insights")
avg_sentiment = df['sentiment'].mean()
total_tweets = len(df)


def get_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['label'] = df['sentiment'].apply(get_label)
most_common_sentiment = df['label'].mode()[0]

st.markdown(f"""
- Total Tweets Analyzed: **{total_tweets}**
- Average Sentiment Score: **{avg_sentiment:.3f}**
- Most Common Sentiment: **{most_common_sentiment}**
""")
