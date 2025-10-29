import os
from typing import List
import torch
from config import config
import threading
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, BloomTokenizerFast, AutoTokenizer, TextIteratorStreamer

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
}

def load_vector_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG['embedding model'],
        model_kwargs={"device": "cuda" if CONFIG['use gpu'] else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(CONFIG['faiss index path'], embedding_model, allow_dangerous_deserialization=True)
    
    return db

def load_llm_model():
    print("Khởi tạo LLM model:", CONFIG['llm model'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['llm model'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['llm model'],
        dtype=CONFIG['dtype'],
        device_map=CONFIG['device map']
    )
    
    return model, tokenizer

def search_context(vector_database, query, k=5):
    retrieved_docs = vector_database.similarity_search_with_score(query, k=k)
    context = ''
    for d, score in retrieved_docs:
        context += f"[{d.metadata['source']}]: {d.page_content}\n"
    return context

def chat(prompt, vector_database, model, tokenizer, max_tokens=256):
    context = search_context(vector_database, prompt, k=CONFIG['top k'])
    
    messages = [
        {"role": "system", "content": f"""
        Bạn là trợ lý AI về dược liệu.
        Quy tắc:
        - Nếu người dùng chào hỏi, hãy đáp lại ngắn gọn, không cần dùng cơ sở tri thức.
        - Nếu người dùng hỏi về dược liệu, hãy tìm trong cơ sở tri thức trước khi trả lời.
        - Nếu không có thông tin nào hữu ích hãy, hãy trả lời: "Xin lỗi, tôi không biết thông tin này."
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
        do_sample=CONFIG['do sample'],
        temperature=CONFIG['temperature'],
        top_p=CONFIG['top p'],
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text 

    thread.join()



if __name__ == "__main__":
    pass