from config import config

# 2. Giờ bạn có thể dùng 'config' như một Python dictionary
def load_llm():
    embedding_model = config.get('embedding_settings', {}).get('embedding_model')
    print(f"Embedding model: {embedding_model}")
    
if __name__ == "__main__":
    print("Chạy llm_model.py trực tiếp để test...")
    load_llm()