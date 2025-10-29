import os
from typing import List
import torch
import pandas as pd
from config import config
from langchain.text_splitter import TextSplitter
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

CONFIG = {
    "embedding model": config.get('embedding_settings', {}).get('embedding_model'),
    "use gpu": torch.cuda.is_available(),
    "file path": 'data/medical plan.csv',
    "faiss index path": 'faiss_index',
    "chunk size": config.get('embedding_settings', {}).get('chunk_size'),
    "chunk overlap": config.get('embedding_settings', {}).get('chunk_overlap'),
}

class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

def convert_to_document(dataframe):
    documents = ''
    mp_name = []

    for index, row in dataframe.iterrows():
        plan = f"{row['Name']} hay còn gọi là {row['Viet_name']} có tên khoa học là {row['Science_name']} thuộc họ {row['Family']} " \
               f"thường được dùng để {row['Uses']}. bộ phận sử dụng của {row['Name']} là {row['Bo_phan_su_dung']}. " \
               f"tính vị của {row['Name']} là {row['Tinh_vi']}. {row['Name']} thường được phân bổ ở {row['Phan_bo_sinh_thai']}."
        text = f'mô tả chi tiết của {row['Name']} là {row['Mo_ta']}. công dụng cụ thể của {row['Name']} là {row['Cong_dung']}'
        mp_name.append(row['Name'].lower())
        mp_name.append(row['Name'].lower())
        documents = documents + plan.lower() + '\n' + text.lower() + '\n'
        
    return documents, mp_name

def create_chunks(documents: str, mp_name: List[str]):
    text_splitter = LineTextSplitter()
    documents_texts = text_splitter.split_text(documents)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CONFIG['chunk size'],
        chunk_overlap=CONFIG['chunk overlap']
    )

    docs: List[Document] = []
    for i, txt in enumerate(documents_texts):
        chunks = text_splitter.split_text(txt)
        for j, c in enumerate(chunks):
            docs.append(Document(
                page_content=c,
                metadata={"source": f"doc_{i}_chunk_{j}_{mp_name[i]}"}
            ))  
    
    print(f"Tạo {len(docs)} chunks từ {len(documents_texts)} documents")
    
    return docs

def load_embedding_model():
    print("Khởi tạo embedding model:", CONFIG['embedding model'])
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG['embedding model'],
        model_kwargs={"device": "cuda" if CONFIG['use gpu'] else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedding_model


if __name__ == "__main__":
    df = pd.read_csv(CONFIG['file path'])
    
    document, mp_name = convert_to_document(df)
    docs = create_chunks(document, mp_name)
    
    embedding_model = load_embedding_model()
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(CONFIG['faiss index path'])
    







