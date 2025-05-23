import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# === Setup Stopwords dan Stemmer ===
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# === Fungsi Praproses Teks ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)
    
# === Load Models ===
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


@st.cache_resource
def load_bert_finetuned():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_fullpraproses")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_fullpraproses")
    return model, tokenizer

@st.cache_resource
def load_bert_pretrained():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_fullpraproses")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_fullpraproses")
    return model, tokenizer

@st.cache_resource
def load_lr_model():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-fullpraproses", filename="lr_model.pkl")
    return joblib.load(file_path)

@st.cache_resource
def load_svm_model():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-fullpraproses", filename="svm_model.pkl")
    return joblib.load(file_path)

# === Prediction Functions ===
def predict_with_bert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
    return pred, probs.numpy()

def predict_with_model(text, model):
    return model.predict([text])[0]

# === Halaman Utama Streamlit ===
st.set_page_config(page_title="Analisis Sentimen Identitas Kependudukan Digital", layout="wide")

menu = st.sidebar.radio("Navigasi", ["Eksplorasi Data", "Prediksi Sentimen"])

if menu == "Eksplorasi Data":
    st.title("Eksplorasi Dataset Ulasan Identitas Kependudukan Digital (IKD)")

    st.markdown("""
    Dataset yang digunakan merupakan ulasan pengguna aplikasi Identitas Kependudukan Digital (IKD) dari Google Play Store.
    Data telah diproses melalui tahapan pembersihan, tokenisasi, stopwords removal, dan stemming.
    """)

    # Load dataset lokal atau Hugging Face jika perlu
    df = pd.read_excel("dataset final (1).xlsx")  

    st.subheader("Contoh Data")
    st.dataframe(df.sample(5))

    st.subheader("Distribusi Sentimen dalam Dataset")
    img1 = load_image_from_url("https://raw.githubusercontent.com/Adkurrr/ikd-streamlit/main/distribusi%20data.png")
    st.image(img1, caption="Distribusi Data Sentimen")

    st.subheader("Wordcloud Kata-Kata Umum")
    img2 = load_image_from_url("https://raw.githubusercontent.com/Adkurrr/ikd-streamlit/main/wordcloud.png")
    st.image(img2, caption="Wordcloud Ulasan")

    st.subheader("Perbandingan Model")
    comparison_data = {
        'Model': ["BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"],
        'Akurasi': [0.95, 0.60, 0.91, 0.91],
        'F1-Score': [0.95, 0.59, 0.91, 0.91],
        'Presisi': [0.95, 0.59, 0.92, 0.91],
        'Recall': [0.95, 0.60, 0.91, 0.91]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

elif menu == "Prediksi Sentimen":
    st.title("Prediksi Sentimen Ulasan IKD")
    st.write("Skenario : Praproses lengkap (Cleansing, Tokenisasi, Stopwords Removal dan Stemming)")

    text_input = st.text_area("Masukkan ulasan:", "")
    model_choice = st.selectbox("Pilih Model", [
        "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"
    ])

    if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_text(text_input)

            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model()
                label = predict_with_model(processed_text, model)
            elif model_choice == "SVM":
                model = load_svm_model()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"

            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")
