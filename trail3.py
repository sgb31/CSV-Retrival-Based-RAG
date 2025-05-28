import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model():
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chatbot = load_model()

# Streamlit UI
st.set_page_config(page_title="Simple RAG with CSV + HF Model")
st.title("📄 Simple RAG App (CSV + Zephyr-7B)")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

# Initialize column descriptions dictionary
column_contexts = {}

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🧾 Step 1: Describe Each Column")
    for col in df.columns:
        column_contexts[col] = st.text_input(f"Describe the column '{col}':", value=f"This column contains {col} data.")

    st.subheader("❓ Step 2: Ask a Question")
    question = st.text_input("Your question about the data:")

    if question and st.button("🔍 Get Answer"):
        docs = df.fillna("").astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()

        def retrieve_relevant_passage(docs, query, top_k=1):
            vectorizer = TfidfVectorizer().fit(docs + [query])
            doc_vectors = vectorizer.transform(docs)
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            return [docs[i] for i in top_indices]

        top_contexts = retrieve_relevant_passage(docs, question, top_k=1)
        context = "\n".join(top_contexts)
        col_desc_text = "\n".join([f"- {col}: {desc}" for col, desc in column_contexts.items()])

        prompt = f"""You are a helpful assistant. Use the following column descriptions and data context to answer the user's question.

Column Descriptions:
{col_desc_text}

Data Context:
{context}

Question:
{question}

Answer:"""

        with st.spinner("Generating answer..."):
            response = chatbot(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                return_full_text=False
            )
            answer = response[0]["generated_text"].strip()

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📌 Retrieved Data Context")
        st.code(context)
