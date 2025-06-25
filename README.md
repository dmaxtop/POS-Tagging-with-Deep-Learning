# POS-Tagging-with-Deep-Learning
Built a POS tagging model using RNN-based architectures (RNN, LSTM, GRU, BiLSTM) on a token-level annotated dataset.
# 🧠 POS Tagging with Deep Learning

This project builds a token-level Part-of-Speech (POS) tagging model using deep learning architectures (RNN, LSTM, GRU, BiLSTM) with a focus on robust preprocessing, embedding integration, and model evaluation.

---

## 📌 Objective

To develop a multi-class classification system that assigns accurate POS tags to each word in a sentence, leveraging sequential neural networks and pre-trained word embeddings (GloVe).

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

- **Dataset**: A CSV-based dataset where each row contains a `Sentence` and a list of corresponding `POS` tags.
- **EDA Highlights**:
  - Inspected null values, data types, and token distributions.
  - Visualized:
    - Sentence length distribution using histograms.
    - Top 20 most frequent words.
    - Frequency of unique POS tags.
- **Key Insight**: The tag distribution was moderately imbalanced, and most sentences ranged between 5–25 tokens.

---

## 🛠 Preprocessing & Feature Engineering

- **Tokenization**: Used Keras `Tokenizer` with `<OOV>` tokens to handle unknown words.
- **Sequence Padding**: Applied `pad_sequences()` to unify input lengths across batches.
- **Label Encoding**: Mapped POS tags to integer class indices using `LabelEncoder`.
- **GloVe Embeddings**: Loaded pre-trained GloVe vectors and used them to initialize the model’s `Embedding` layer for better semantic understanding.

---

## 🤖 Model Architectures

Implemented and compared four neural network models:

| Model         | Layers Used                                  |
|---------------|----------------------------------------------|
| Simple RNN    | Embedding → Masking → SimpleRNN → Dense     |
| LSTM          | Embedding → Masking → LSTM → Dense          |
| GRU           | Embedding → Masking → GRU → Dense           |
| BiLSTM        | Embedding → Masking → BiLSTM → Dense        |

- All models used `TimeDistributed(Dense)` to predict POS tags at the token level.
- `Softmax` activation for multi-class output per token.
- `Dropout` was applied to prevent overfitting.

---

## 📈 Evaluation Metrics

Each model was evaluated on:
- **Accuracy**
- **Macro F1-score**
- **Weighted F1-score**
- **Confusion Matrix**
- **Classification Report**

Evaluation was done on both training and test sets using `scikit-learn`.

---

## 🧪 Results & Insights

- **LSTM and BiLSTM** showed superior performance over vanilla RNN and GRU models.
- **BiLSTM** performed best due to its ability to capture both forward and backward context.
- **GloVe embeddings** significantly improved model performance, especially for low-frequency tokens.
- Balanced accuracy and F1-score showed the model was robust across all classes.

---

## 📦 Tools & Libraries

- Python (3.8+)
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- GloVe word vectors (100d)

---

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
