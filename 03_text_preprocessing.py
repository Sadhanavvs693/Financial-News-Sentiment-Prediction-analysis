import pandas as pd
import re
import numpy as np
from collections import Counter

# Load training and validation datasets
train_df = pd.read_csv("data/sent_train.csv")
valid_df = pd.read_csv("data/sent_valid.csv")

# Display sample tweets
print("Original Training Tweets:")
print(train_df["text"].head())

print("\nOriginal Validation Tweets:")
print(valid_df["text"].head())

import re

# Function to remove URLs
def remove_urls(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return text

# Apply URL removal
train_df["cleaned_text"] = train_df["text"].apply(remove_urls)
valid_df["cleaned_text"] = valid_df["text"].apply(remove_urls)

# Compare original and cleaned tweets
print("\nOriginal Tweet:")
print(train_df["text"].iloc[0])

print("\nCleaned Tweet:")
print(train_df["cleaned_text"].iloc[0])

# Function to remove extra spaces
def remove_extra_spaces(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply extra space removal
train_df["cleaned_text"] = train_df["cleaned_text"].apply(remove_extra_spaces)
valid_df["cleaned_text"] = valid_df["cleaned_text"].apply(remove_extra_spaces)

print("\nText After Removing Extra Spaces:")
print(train_df["cleaned_text"].iloc[0])

# Function to remove Twitter mentions
def remove_mentions(text):
    text = re.sub(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+", "", text)
    return text

# Apply mention removal
train_df["cleaned_text"] = train_df["cleaned_text"].apply(remove_mentions)
valid_df["cleaned_text"] = valid_df["cleaned_text"].apply(remove_mentions)

# Remove extra spaces again
train_df["cleaned_text"] = train_df["cleaned_text"].apply(remove_extra_spaces)
valid_df["cleaned_text"] = valid_df["cleaned_text"].apply(remove_extra_spaces)

print("\nText After Removing Mentions:")
print(train_df["cleaned_text"].iloc[0])

# Function to convert text to lowercase
def convert_lowercase(text):
    return text.lower()

# Apply lowercase conversion
train_df["cleaned_text"] = train_df["cleaned_text"].apply(convert_lowercase)
valid_df["cleaned_text"] = valid_df["cleaned_text"].apply(convert_lowercase)

print("\nText After Lowercase Conversion:")
print(train_df["cleaned_text"].iloc[0])

# Check preprocessing results

print("\nOriginal Dataset Shape:")
print("Training:", train_df.shape)
print("Validation:", valid_df.shape)

print("\nMissing Values in Cleaned Text:")
print("Training:", train_df["cleaned_text"].isnull().sum())
print("Validation:", valid_df["cleaned_text"].isnull().sum())

print("\nSample Original and Cleaned Tweets:")

for i in range(5):
    print("\nOriginal:", train_df["text"].iloc[i])
    print("Cleaned:", train_df["cleaned_text"].iloc[i])

    # Check empty cleaned tweets

print("\nEmpty Cleaned Tweets:")

print(
    "Training:",
    (train_df["cleaned_text"].str.strip() == "").sum()
)

print(
    "Validation:",
    (valid_df["cleaned_text"].str.strip() == "").sum()
)

# Display tweets that became empty after cleaning

empty_tweets = train_df[
    train_df["cleaned_text"].str.strip() == ""
]

print("\nOriginal tweets that became empty:")

print(empty_tweets[["text", "label", "cleaned_text"]].to_string(index=False))

# Remove empty cleaned tweets

train_df = train_df[
    train_df["cleaned_text"].str.strip() != ""
].copy()

valid_df = valid_df[
    valid_df["cleaned_text"].str.strip() != ""
].copy()

print("\nDataset Shape After Removing Empty Tweets:")
print("Training:", train_df.shape)
print("Validation:", valid_df.shape)

print("\nRemaining Empty Tweets:")
print("Training:", (train_df["cleaned_text"].str.strip() == "").sum())
print("Validation:", (valid_df["cleaned_text"].str.strip() == "").sum())

# Tokenization

def tokenize_text(text):
    tokens = re.findall(
        r":\)|:\(|\$?[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?",
        text.lower()
    )
    return tokens


train_df["tokens"] = train_df["cleaned_text"].apply(tokenize_text)
valid_df["tokens"] = valid_df["cleaned_text"].apply(tokenize_text)


# Display tokenization results

print("\nTokenization Results:")

for i in range(5):
    print("\nCleaned Text:", train_df["cleaned_text"].iloc[i])
    print("Tokens:", train_df["tokens"].iloc[i])

    from collections import Counter

# Build vocabulary using training tokens only

word_counts = Counter(
    word
    for tokens in train_df["tokens"]
    for word in tokens
)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in word_counts:
    vocab[word] = len(vocab)

print("\nVocabulary Size:", len(vocab))

print("\nFirst 20 Vocabulary Entries:")

for word, index in list(vocab.items())[:20]:
    print(word, ":", index)

    # Convert tokens into numerical sequences

def convert_to_sequences(tokens):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokens]


train_df["sequences"] = train_df["tokens"].apply(convert_to_sequences)
valid_df["sequences"] = valid_df["tokens"].apply(convert_to_sequences)


# Display numerical sequences

print("\nNumerical Sequence Results:")

for i in range(5):
    print("\nTokens:", train_df["tokens"].iloc[i])
    print("Sequence:", train_df["sequences"].iloc[i])

    # Calculate sequence lengths

train_df["sequence_length"] = train_df["sequences"].apply(len)

print("\nSequence Length Statistics:")
print(train_df["sequence_length"].describe())

print("\nMaximum Sequence Length:")
print(train_df["sequence_length"].max())

print("\n95th Percentile Sequence Length:")
print(train_df["sequence_length"].quantile(0.95))

# Check tweets with zero tokens

empty_token_tweets = train_df[
    train_df["sequences"].apply(len) == 0
]

print("\nTweets with Zero Tokens:")
print("Count:", len(empty_token_tweets))

print("\nOriginal Tweets:")
print(
    empty_token_tweets[
        ["text", "cleaned_text", "label"]
    ].to_string(index=False)
)
# Padding sequences

MAX_LEN = 20

def pad_sequences_manual(sequences, max_len):
    padded = np.zeros(
        (len(sequences), max_len),
        dtype=np.int32
    )

    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_len]
        padded[i, :len(sequence)] = sequence

    return padded


# Apply padding

X_train = pad_sequences_manual(
    train_df["sequences"].tolist(), MAX_LEN
)

X_valid = pad_sequences_manual(
    valid_df["sequences"].tolist(), MAX_LEN
)


# Display results

print("\nPadded Sequence Shapes:")
print("Training:", X_train.shape)
print("Validation:", X_valid.shape)

print("\nFirst Padded Sequence:")
print(X_train[0])

print("\nSecond Padded Sequence:")
print(X_train[1])
# Prepare sentiment labels

y_train = train_df["label"].to_numpy(dtype=np.int64)
y_valid = valid_df["label"].to_numpy(dtype=np.int64)


# Display label shapes

print("\nLabel Shapes:")
print("Training:", y_train.shape)
print("Validation:", y_valid.shape)


# Display unique labels

print("\nUnique Labels:")
print(np.unique(y_train))
print(np.unique(y_valid))


# Display first 10 labels

print("\nFirst 10 Training Labels:")
print(y_train[:10])
# Final sentiment distribution

print("\nFinal Training Sentiment Distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nFinal Validation Sentiment Distribution:")
print(valid_df["label"].value_counts().sort_index())
import os
import json

# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)

# Save training and validation data
np.savez_compressed(
    "artifacts/preprocessed_data.npz",
    X_train=X_train,
    X_valid=X_valid,
    y_train=y_train,
    y_valid=y_valid
)

# Save vocabulary
with open("artifacts/vocabulary.json", "w") as file:
    json.dump(vocab, file)

print("\nPreprocessed data saved successfully!")
print("Vocabulary saved successfully!")