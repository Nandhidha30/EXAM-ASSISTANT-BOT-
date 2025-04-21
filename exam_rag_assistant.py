import os
import fitz  
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

DOCS_PATH = "./docs"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_FILE = "rag_data.pkl"
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_faiss_index():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = []
    sources = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith('.pdf'):
            content = extract_text_from_pdf(os.path.join(DOCS_PATH, file))
            chunks = split_text(content)
            texts.extend(chunks)
            sources.extend([file] * len(chunks))

    embeddings = embedder.encode(texts, convert_to_tensor=False)
    embeddings_np = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, texts, sources, embedder), f)

    print(" FAISS index built and saved.")

def load_rag_components():
    with open(INDEX_FILE, "rb") as f:
        index, texts, sources, embedder = pickle.load(f)
    return index, texts, sources, embedder

def get_relevant_chunks(query, index, texts, sources, embedder, top_k=5):
    q_embed = embedder.encode([query])[0].astype("float32")
    D, I = index.search(np.array([q_embed]), top_k)
    return [(texts[i], sources[i]) for i in I[0]]

def generate_answer(query, index, texts, sources, embedder):
    context_chunks = get_relevant_chunks(query, index, texts, sources, embedder)
    context_text = "\n\n".join([f"From {src}:\n{text}" for text, src in context_chunks])

    prompt = f"""You are an expert tutor for competitive exams (UPSC/JEE/NEET).\nUse the following context to answer the question.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def generate_mcq(text_chunk):
    prompt = f"Generate 3 multiple-choice questions from the following study content:\n\n{text_chunk}\n\nInclude options and mark the correct answer."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def summarize_topic(text_chunk):
    prompt = f"Summarize the following content in bullet points:\n\n{text_chunk}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help="Build the FAISS index")
    parser.add_argument('--query', type=str, help="Ask a question")
    parser.add_argument('--quiz', type=str, help="Generate quiz from topic")
    parser.add_argument('--summary', type=str, help="Summarize a topic")
    args = parser.parse_args()

    if args.build:
        build_faiss_index()
    else:
        index, texts, sources, embedder = load_rag_components()
        if args.query:
            ans = generate_answer(args.query, index, texts, sources, embedder)
            print("\n Answer:\n", ans)
        elif args.quiz:
            print("\n MCQs:\n", generate_mcq(args.quiz))
        elif args.summary:
            print("\n Summary:\n", summarize_topic(args.summary))
