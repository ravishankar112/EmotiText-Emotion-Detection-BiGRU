# EmotiText: Emotion Detection from Text

A deep learning web app that reads a sentence and predicts which of six emotions it expresses:
**sadness, joy, love, anger, fear, surprise**. It uses a **Bidirectional GRU (BiGRU)** model built with
TensorFlow/Keras and a Streamlit frontend.

**Developer and designer:** Ravi Shankar Kumar (Data Science and ML)

## Features
- Predicts the main emotion with a confidence score
- Shows the score for all 6 emotions as bars
- One-click example sentences
- Recent checks history
- Dark, responsive interface

## Tech stack
Python, TensorFlow, Keras, Streamlit, NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Hugging Face Datasets

## Model details
| Item | Value |
|---|---|
| Dataset | `dair-ai/emotion` (Hugging Face) |
| Preprocessing | Keras Tokenizer (10,000 words, OOV token), padding to 50 tokens |
| Models compared | RNN, LSTM, GRU, BiGRU, BiLSTM |
| Final model | 2 x Bidirectional GRU, 300-dim embedding, dropout 0.5 |
| Imbalance handling | Balanced class weights |
| Training control | Early stopping on validation loss |

## Project structure
```
EmotiText/
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── main.ipynb              # Training notebook
├── .streamlit/
│   └── config.toml         # Dark theme
└── Artifacts/
    ├── BiGRU_Model.keras   # Trained model
    └── tokenizer.pkl       # Fitted tokenizer
```

## Run locally
```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. install packages
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```
Open http://localhost:8501 in your browser.

> **Important:** install the same TensorFlow version you used for training. Check it with `pip show tensorflow`
> and put it in `requirements.txt` (example: `tensorflow==2.16.1`). A version mismatch is the most common
> reason a `.keras` model fails to load.

## Deploy for free
**Streamlit Community Cloud**
1. Push this folder to a GitHub repository (the model is about 40 MB, which is fine for GitHub).
2. Go to share.streamlit.io, connect the repo and select `app.py`.

**Hugging Face Spaces:** create a new Space, choose Streamlit, upload the same files.

## Limitations
- Trained on short English sentences; other languages and long paragraphs will be less accurate.
- Only six emotions are supported, so mixed or neutral text is forced into the closest one.

## Author
**Ravi Shankar Kumar**, Data Science and ML
