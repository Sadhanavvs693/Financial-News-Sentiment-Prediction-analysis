import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training dataset
train_df = pd.read_csv("data/sent_train.csv")
valid_df = pd.read_csv("data/sent_valid.csv")

# 1. Sentiment distribution
sentiment_names = {
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}

sentiment_counts = train_df["label"].value_counts().sort_index()
sentiment_counts.index = sentiment_counts.index.map(sentiment_names)

print("Sentiment Distribution:")
print(sentiment_counts)

# 2. Text length analysis
train_df["text_length"] = train_df["text"].apply(len)

print("\nText Length Statistics:")
print(train_df["text_length"].describe())

# 3. Plot text length distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x="text_length",
    bins=30,
    kde=True
)

plt.title("Distribution of Tweet Text Length")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")

plt.show()

# Word count analysis
train_df["word_count"] = train_df["text"].apply(
    lambda text: len(text.split())
)

print("\nWord Count Statistics:")
print(train_df["word_count"].describe())

# Plot word count distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x="word_count",
    bins=30,
    kde=True
)

plt.title("Distribution of Word Count")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.show()

# Word count analysis
train_df["word_count"] = train_df["text"].apply(
    lambda text: len(text.split())
)


# Plot word count distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=train_df,
    x="word_count",
    bins=30,
    kde=True
)
plt.title("Distribution of Word Count")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# Average word count for each sentiment
sentiment_names = {
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}

avg_word_count = train_df.groupby("label")["word_count"].mean()

avg_word_count.index = avg_word_count.index.map(sentiment_names)

print("\nAverage Word Count by Sentiment:")
print(avg_word_count)

# Sentiment distribution bar chart
sentiment_counts = train_df["label"].map(sentiment_names).value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.show()

from collections import Counter
import re

# Combine all tweet text
all_text = " ".join(train_df["text"].astype(str))

# Convert text to lowercase
all_text = all_text.lower()

# Remove URLs
all_text = re.sub(r"https?://\S+|www\.\S+", "", all_text)

# Extract words
words = re.findall(r"\b[a-z]+\b", all_text)

# Count word frequency
word_counts = Counter(words)

# Display top 20 words
print("\nTop 20 Most Frequent Words After URL Removal:")

for word, count in word_counts.most_common(20):
    print(word, ":", count)

    import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Download English stop words
nltk.download("stopwords")

# Get English stop words
stop_words = set(stopwords.words("english"))

# Combine tweet text
all_text = " ".join(train_df["text"].astype(str)).lower()

# Remove URLs
all_text = re.sub(r"https?://\S+|www\.\S+", "", all_text)

# Extract words
words = re.findall(r"\b[a-z]+\b", all_text)

# Remove stop words
filtered_words = [
    word for word in words if word not in stop_words
]

# Count word frequency
word_counts = Counter(filtered_words)

print("\nTop 20 Words After Stop Word Removal:")

for word, count in word_counts.most_common(20):
    print(word, ":", count)

    # Top 20 frequent words visualization
top_words = word_counts.most_common(20)

words_list = [item[0] for item in top_words]
counts_list = [item[1] for item in top_words]

plt.figure(figsize=(10, 7))

sns.barplot(
    x=counts_list,
    y=words_list
)

plt.title("Top 20 Most Frequent Financial Words")
plt.xlabel("Frequency")
plt.ylabel("Words")

plt.tight_layout()
plt.show()

# Check duplicate tweets across training and validation datasets

train_texts = set(train_df["text"].astype(str))
valid_texts = set(valid_df["text"].astype(str))

common_texts = train_texts.intersection(valid_texts)

print("\nDuplicate Tweets Across Datasets:")
print("Number of common tweets:", len(common_texts))