import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  

# Set page config
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #495057;
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4a4e69;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #22223b;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .title {
            color: #22223b;
            text-align: center;
            margin-bottom: 30px;
        }
        .result {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #4a4e69;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# Loading the model
@st.cache_resource
def load_my_model():
    model = load_model('next_word_lstm.h5')
    return model

model = load_my_model()

# Load the tokenizer
@st.cache_resource
def load_my_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_my_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len-1):]  # Take the last max_sequence_len-1 tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None            

# App layout
st.markdown("<h1 class='title'>🔮 Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: #6c757d; margin-bottom: 30px;'>
        A sophisticated LSTM model that predicts the next word in your sentence.
    </p>
""", unsafe_allow_html=True)

# Main container
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_input(
            "Enter your text here:",
            "To be or not to be",
            key="input_text",
            help="Type a sentence and let the AI predict the next word"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        predict_btn = st.button("Predict Next Word", key="predict_btn")

max_sequence_len = 14  # Replace with your actual sequence length used during training

if predict_btn:
    with st.spinner('🔍 Predicting the next word...'):
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.markdown(f"""
            <div class='result'>
                <h3>Prediction Result</h3>
                <p style='font-size: 1.2em;'>Input text: <strong>{input_text}</strong></p>
                <p style='font-size: 1.5em; color: #4a4e69;'>Next word: <strong style='color: #22223b;'>{next_word}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Could not predict the next word. Please try different input.")

# Footer
st.markdown("""
    <div class='footer'>
        <hr style='border: 0.5px solid #dee2e6; margin: 20px 0;'>
        <p>Powered by LSTM with Early Stopping • Made with Streamlit</p>
    </div>
""", unsafe_allow_html=True)