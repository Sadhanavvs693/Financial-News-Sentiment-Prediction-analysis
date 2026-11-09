# Financial News Sentiment Prediction Using Deep Learning

## 1. Project Overview

This project uses Deep Learning and Natural Language Processing (NLP) to classify financial news text into three sentiment categories:

- **Bearish (0):** Negative financial sentiment.
- **Bullish (1):** Positive financial sentiment.
- **Neutral (2):** Neither clearly positive nor negative.

The objective is to identify sentiment in financial news and demonstrate how deep learning models can be used for financial text classification.

## 2. Dataset

The project uses the Twitter Financial News Sentiment dataset.

The supplied CSV files contain:

| Dataset | Records |
|---|---:|
| Training | 9,543 |
| Validation | 2,388 |

The training dataset contains three sentiment classes:

| Sentiment | Records |
|---|---:|
| Bearish | 1,442 |
| Bullish | 1,923 |
| Neutral | 6,178 |

The dataset has an imbalanced class distribution, with Neutral as the majority class.

**Data note:** The supplied project brief reports different dataset counts from the provided CSV files. This implementation uses the actual supplied CSV files and their observed record counts.

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Removed URLs and user mentions.
2. Removed extra spaces.
3. Converted text to lowercase.
4. Tokenized the text.
5. Built a vocabulary using training data.
6. Converted tokens into numerical sequences.
7. Applied padding and truncation to a maximum sequence length of 20.
8. Converted the labels into numerical format.

Financial terms and stock tickers were preserved during preprocessing.

Six training records that contained no meaningful text after cleaning were removed.

## 4. Deep Learning Models

Three baseline deep learning architectures were implemented using PyTorch:

- **Simple RNN:** Processes sequential text using recurrent connections.
- **LSTM:** Uses memory cells and gates to capture longer-term dependencies.
- **GRU:** Uses update and reset gates to process sequential information.

### Model configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 64 |
| Hidden dimension | 128 |
| Number of classes | 3 |
| Maximum sequence length | 20 |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |

## 5. Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score and confusion matrices.

### Model comparison

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Simple RNN | 72.91% | 0.6008 |
| LSTM | 75.29% | 0.6638 |
| GRU | 76.68% | 0.6743 |

GRU achieved the highest accuracy and macro F1-score among the three reported evaluation runs.

Macro F1-score is particularly relevant because the dataset has an imbalanced class distribution.

**Evaluation note:** The comparison metrics were obtained from separate evaluation runs. The model checkpoints were subsequently saved through additional training runs, so the reported metrics should not be assumed to describe those exact saved checkpoints.

## 6. Streamlit Deployment

A Streamlit application was developed to demonstrate sentiment prediction.

Application features:

- Select Simple RNN, LSTM or GRU.
- Enter financial news text.
- Preprocess and tokenize the input.
- Generate sentiment predictions.
- Display the predicted class and confidence.

The application was tested using three illustrative inputs. The GRU model returned the expected interpretations for all three examples.

These tests demonstrate application functionality and do not represent overall model accuracy.

## 7. Project Structure

```text
Financial_News_Sentiment_Prediction/
│
├── artifacts/
│   ├── preprocessed_data.npz
│   ├── vocabulary.json
│   ├── model_accuracy_comparison.png
│   └── model_f1_comparison.png
│
├── data/
│   ├── sent_train.csv
│   ├── sent_valid.csv
│   └── sent_valid.xlsx
│
├── models/
│   ├── rnn_model.pth
│   ├── lstm_model.pth
│   └── gru_model.pth
│
├── 01_dataset_loading.py
├── 02_eda.py
├── 03_text_preprocessing.py
├── 04_rnn_model.py
├── 05_lstm_model.py
├── 06_gru_model.py
├── 07_model_comparison.py
├── app.py
└── README.md
```

## 8. Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit

## 9. How to Run the Application

1. Activate the project virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

2. Install the required packages if they are not already installed.

3. Run the Streamlit application:

```powershell
python -m streamlit run app.py
```

4. Open the local URL displayed in the terminal.

## 10. Conclusion

This project demonstrates the application of recurrent deep learning architectures to financial