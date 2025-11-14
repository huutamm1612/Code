import os
from typing import List
import numpy as np
import pandas as pd
import torch
from config import config
import threading
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, BloomTokenizerFast, AutoTokenizer, TextIteratorStreamer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi

db = None
embedding_model = None
df = None
docs = None
bm25 = None

CONFIG = {
    "embedding model": config.get('embedding_settings', {}).get('embedding_model'),
    "llm model": config.get('llm_settings', {}).get('llm_model'),
    "use gpu": torch.cuda.is_available(),
    "faiss index path": 'faiss_index',
    "device map": "auto",
    "dtype": torch.bfloat16,
    "top k": config.get('llm_settings', {}).get('top_k'),
    "top p": config.get('llm_settings', {}).get('top_p'),
    "temperature": config.get('llm_settings', {}).get('temperature'),
    "max tokens": config.get('llm_settings', {}).get('max_tokens'),
    "do sample": True,
    "file path": config.get('paths', {}).get('medical_info_path')
}

query_templates = [
    "[herb] có tác dụng gì",
    "[herb] chữa được bệnh gì",
    "[herb] là cây gì",
    "[herb] dùng bộ phận nào",
    "[herb] mọc ở đâu",
    "[herb] có độc không",
    "[herb] có tác hại gì",
    "[herb] chế biến như thế nào",
    "[herb] có ăn được không",
]

def load_vector_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG['embedding model'],
        model_kwargs={"device": "cuda" if CONFIG['use gpu'] else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(CONFIG['faiss index path'], embedding_model, allow_dangerous_deserialization=True)
    
    return db, embedding_model

def load_llm_model():
    print("Khởi tạo LLM model:", CONFIG['llm model'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['llm model'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['llm model'],
        dtype=CONFIG['dtype'],
        device_map=CONFIG['device map']
    )
    
    return model, tokenizer

def extract_entity(text, threshold=80):
    best_match = None
    best_score = 0

    for herb in df['Name']:
        score = fuzz.partial_ratio(herb.lower(), text.lower())
        if score > best_score:
            best_match = herb
            best_score = score

    return best_match if best_score >= threshold else None

def unify_query(query, templates, embedding_model, threshold=0.8):
    herb = extract_entity(query)
    if not herb:
        return query, None, None

    stripped_query = query.lower().replace(herb.lower(), "[herb]")

    template_embeddings = embedding_model.embed_documents(templates)
    query_embedding = embedding_model.embed_query(stripped_query)

    sims = cosine_similarity([query_embedding], template_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= threshold:
        unified = templates[best_idx].replace("[herb]", herb)
    else:
        unified = query

    return unified, herb, best_score

def hybrid_search(query, alpha=0.6, k_sem=10, k_kw=10):
    sem_results = db.similarity_search_with_score(query, k=k_sem)
    sem_scores = {d.page_content: score for d, score in sem_results}

    bm25_scores = bm25.get_scores(query.split())
    top_kw_idx = np.argsort(bm25_scores)[::-1][:k_kw]
    kw_results = [docs[i] for i in top_kw_idx]
    kw_scores = {docs[i].page_content: bm25_scores[i] for i in top_kw_idx}

    combined = {}
    for d in set(list(sem_scores.keys()) + list(kw_scores.keys())):
        sem = sem_scores.get(d, 0)
        kw = kw_scores.get(d, 0)
        combined[d] = alpha * sem + (1 - alpha) * kw

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [(next(doc for doc in docs if doc.page_content == d), score) for d, score in ranked]

def search_context(query, alpha=0.6, k_sem=10, k_kw=10, top_k=5):
    hybrid_docs = hybrid_search(query, alpha, k_sem, k_kw)
    selected = hybrid_docs[:top_k]

    context = ""
    for i, (d, score) in enumerate(selected, start=1):
        context += f"[{d.metadata['source']}]: {d.page_content}\n"
    return context

def chat(prompt, vector_database, model, tokenizer, max_tokens=256):
    if embedding_model is None:
        db, embedding_model = load_vector_database()
        docs = db.get_all_documents()
        df = pd.read_csv(CONFIG['file path'])
        bm25 = BM25Okapi([d.page_content.split() for d in docs])
        
    
    prompt, _, _ = unify_query(prompt, query_templates, embedding_model)
    context = search_context(vector_database, prompt, k=CONFIG['top k'])
    
    messages = [
        {"role": "system", "content": f"""
        Bạn là trợ lý AI về dược liệu.
        Quy tắc:
        - Nếu người dùng chào hỏi, hãy đáp lại ngắn gọn, không cần dùng cơ sở tri thức.
        - Nếu người dùng hỏi về dược liệu, hãy tìm trong cơ sở tri thức trước khi trả lời.
        - Nếu không có thông tin nào hữu ích hoặc không liên quan đến dược liệu hãy, hãy trả lời: "Xin lỗi, tôi không biết thông tin này."
        Hãy trả lời dựa trên các thông tin sau:

        [Ngữ cảnh từ cơ sở tri thức]
        {context}
        """},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

if __name__ == "__main__":
    db, embedding_model = load_vector_database()
    docs = db.get_all_documents()
    df = pd.read_csv(CONFIG['file path'])
    bm25_corpus = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(bm25_corpus)
    
    
    