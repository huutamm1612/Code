from flask import Flask, request, Response, send_from_directory, jsonify
from model.RAG import load_vector_database, load_llm_model, chat

app = Flask(__name__, static_folder="static")

print("Đang tải FAISS database và mô hình LLM...")
vector_db = load_vector_database()
model, tokenizer = load_llm_model()
print("✅ Hệ thống sẵn sàng!")

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

def generate(prompt):
    for token in chat(prompt, vector_db, model, tokenizer):
        yield token
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    prompt = data.get("prompt", "")
    print(prompt)
    if not prompt.strip():
        return jsonify({"error": "empty prompt"}), 400

    print("New chat request processed.")
    return Response(generate(prompt), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
