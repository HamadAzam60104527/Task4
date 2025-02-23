import streamlit as st
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')

documents = [reuters.raw(fileid) for fileid in reuters.fileids()]

def preprocess(text):
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

preprocessed_documents = [preprocess(doc) for doc in documents]

word2vec_model = Word2Vec(sentences=preprocessed_documents, vector_size=100, window=5, min_count=1, workers=4)

def compute_average_embedding(text, model):

    words = preprocess(text)
    embeddings = [model.wv[word] for word in words if word in model.wv]

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

document_embeddings = [compute_average_embedding(" ".join(doc), word2vec_model) for doc in preprocessed_documents]

st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

if st.button("Search"):

    query_embedding = compute_average_embedding(query, word2vec_model)
    similarities = [cosine_similarity([query_embedding], [doc_embedding])[0][0] for doc_embedding in document_embeddings]

    N = 5

    top_n_indices = np.argsort(similarities)[-N:][::-1]

    st.write(f"### Top {N} Most Relevant Documents:")

    for i in top_n_indices:
        st.write(f"- **Document {i}** (Score: {similarities[i]:.4f})")
