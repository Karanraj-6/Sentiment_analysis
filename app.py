import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import boto3
import os

# aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
# aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
# region = st.secrets["AWS_DEFAULT_REGION"]

# # Initialize S3 client with credentials
# s3 = boto3.client(
#     's3',
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key,
#     region_name=region
# )
# bucket_name = 'sentimentanalysiskaran'  # Your S3 bucket name

# # List of model files to download from S3
# files = [
#     {"s3_key": "sentiment_model/config.json", "local_path": "sentiment_model/config.json"},
#     {"s3_key": "sentiment_model/model.safetensors", "local_path": "sentiment_model/model.safetensors"},
#     {"s3_key": "sentiment_model/special_tokens_map.json", "local_path": "sentiment_model/special_tokens_map.json"},
#     {"s3_key": "sentiment_model/tokenizer_config.json", "local_path": "sentiment_model/tokenizer_config.json"},
#     {"s3_key": "sentiment_model/vocab.txt", "local_path": "sentiment_model/vocab.txt"}
# ]

# # Ensure directory exists for model files
# if not os.path.exists('sentiment_model'):
#     os.makedirs('sentiment_model')

# # Function to download files from S3
# def download_from_s3():
#     """Download model files from S3."""
#     for file in files:
#         try:
#             print(f"Downloading {file['s3_key']} from S3...")
#             s3.download_file(bucket_name, file['s3_key'], file['local_path'])
#             print(f"{file['s3_key']} downloaded successfully.")
#         except Exception as e:
#             print(f"Error downloading {file['s3_key']}: {str(e)}")
#             st.error(f"Error downloading {file['s3_key']}: {str(e)}")
#             st.stop()  # Stop execution if download fails

# # Download model files
# download_from_s3()

# Load the model and tokenizer
try:
    model = DistilBertForSequenceClassification.from_pretrained('./sentiment_model')
    tokenizer = DistilBertTokenizer.from_pretrained('./sentiment_model')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()  # Stop execution if model loading fails

def preprocess_input(input_text):
    """Preprocess the input text for the model."""
    encoded_input = tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    return encoded_input

# Streamlit UI elements
st.markdown(
    """
    <style>
    .custom-text-area {
        border: 3px solid;
        border-image: linear-gradient(to right, red, yellow, green, blue) 1;
        border-radius: 5px;
        padding: 10px;
        height: 200px;  /* Adjust height as needed */
        resize: none;   /* Disable resizing */
        width: 100%;    /* Full width */
        box-sizing: border-box; /* Ensure padding does not affect overall width */
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='color: black;'>Sentiment Analysis with BERT</h1>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your text:", 
    height=200, 
    key="input_text", 
    help="Type your text here...", 
    placeholder="Type your text here..."
)

st.markdown(
    '<style>div[data-baseweb="textarea"] { border: 3px solid; border-image: linear-gradient(to right, red, yellow, green, blue) 1; border-radius: 5px; padding: 10px;}</style>', 
    unsafe_allow_html=True
)

if st.button("Analyze"):
    if user_input:  
        preprocessed_input = preprocess_input(user_input)
        
        with torch.no_grad():
            outputs = model(**preprocessed_input)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        sentiment = "Positive" if predictions.item() == 1 else "Negative"
        
        if sentiment == "Positive":
            st.markdown(f"<h1 style='color: green;'>{sentiment}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: red;'>{sentiment}</h1>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text.")