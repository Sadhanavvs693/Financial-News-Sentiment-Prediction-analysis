import streamlit as st
import torch
import torch.nn as nn
import json
import re
import numpy as np

# Load vocabulary
with open("artifacts/vocabulary.json", "r") as file:
    vocab = json.load(file)

vocab_size = len(vocab)

# Sentiment labels
label_names = {
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}

MAX_LEN = 20

st.title("Financial News Sentiment Prediction")
st.write("Enter financial news text to predict its sentiment.")

st.success("Vocabulary loaded successfully!")
st.write("Vocabulary size:", vocab_size)
# Simple RNN Architecture
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleRNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        lengths = (x != 0).sum(dim=1)

        last_output = output[
            torch.arange(x.size(0)),
            lengths - 1
        ]

        return self.fc(last_output)


# LSTM Architecture
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)

        lengths = (x != 0).sum(dim=1)

        last_output = output[
            torch.arange(x.size(0)),
            lengths - 1
        ]

        return self.fc(last_output)


# GRU Architecture
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(GRUModel, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)

        lengths = (x != 0).sum(dim=1)

        last_output = output[
            torch.arange(x.size(0)),
            lengths - 1
        ]

        return self.fc(last_output)
    # Model parameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 3

# Select CPU for prediction
device = torch.device("cpu")


# Create and load Simple RNN
rnn_model = SimpleRNN(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES
)
rnn_model.load_state_dict(
    torch.load("models/rnn_model.pth", map_location=device, weights_only=True)
)
rnn_model.eval()


# Create and load LSTM
lstm_model = LSTMModel(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES
)
lstm_model.load_state_dict(
    torch.load("models/lstm_model.pth", map_location=device, weights_only=True)
)
lstm_model.eval()


# Create and load GRU
gru_model = GRUModel(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES
)
gru_model.load_state_dict(
    torch.load("models/gru_model.pth", map_location=device, weights_only=True)
)
gru_model.eval()

st.success("All three trained models loaded successfully!")

# Clean input text
def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(
        r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+", "", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    return text


# Tokenize input text
def tokenize_text(text):
    tokens = re.findall(
        r":\)|:\(|\$?[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?",
        text.lower()
    )
    return tokens


# Convert tokens into numerical sequences
def convert_to_sequence(tokens):
    return [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]


# Pad sequence to maximum length
def pad_sequence(sequence):
    padded = np.zeros((1, MAX_LEN), dtype=np.int64)

    sequence = sequence[:MAX_LEN]

    padded[0, :len(sequence)] = sequence

    return torch.tensor(padded, dtype=torch.long)

# Generate sentiment prediction
def predict_sentiment(model, text):

    cleaned_text = clean_text(text)

    tokens = tokenize_text(cleaned_text)

    if len(tokens) == 0:
        return None, None

    sequence = convert_to_sequence(tokens)

    input_tensor = pad_sequence(sequence)

    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(
            probabilities, dim=1
        ).item()

        confidence = probabilities[0][predicted_class].item()

    predicted_sentiment = label_names[predicted_class]

    return predicted_sentiment, confidence

# Model selection
st.subheader("Choose a Deep Learning Model")

model_choice = st.selectbox(
    "Select Model",
    ["GRU", "LSTM", "Simple RNN"]
)

# Financial news input
st.subheader("Enter Financial News")

user_input = st.text_area(
    "News Text",
    placeholder="Example: The company reported strong quarterly earnings and revenue growth."
)

# Prediction button
if st.button("Predict Sentiment"):

    if not user_input.strip():
        st.warning("Please enter some financial news text.")

    else:
        if model_choice == "GRU":
            selected_model = gru_model

        elif model_choice == "LSTM":
            selected_model = lstm_model

        else:
            selected_model = rnn_model

        sentiment, confidence = predict_sentiment(
            selected_model, user_input
        )

        if sentiment is None:
            st.warning("Please enter meaningful financial news text.")

        else:
            st.subheader("Prediction Result")

            if sentiment == "Bullish":
                st.success(f"Sentiment: {sentiment}")

            elif sentiment == "Bearish":
                st.error(f"Sentiment: {sentiment}")

            else:
                st.info(f"Sentiment: {sentiment}")

            st.write(
                f"Prediction Confidence: {confidence * 100:.2f}%"
            )