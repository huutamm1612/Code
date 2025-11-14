from flask import Flask, request, Response, send_from_directory, jsonify
from src.model.RAG import load_vector_database, load_llm_model, chat
from src.model.classification import load_classification_model
import torch
from PIL import Image
from torchvision import transforms

app = Flask(__name__, static_folder="static")

print("Đang tải các model cần thiết...")
cls_model, test_transform, idx_to_class = load_classification_model()
vector_db, embedding_model = load_vector_database()
model, tokenizer = load_llm_model()
print("Hệ thống sẵn sàng!")

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
    prompt = request.form.get("prompt", "").strip()
    image_file = request.files.get("image")

    if not prompt and not image_file:
        return Response("Lỗi: Không có nội dung hoặc ảnh!", status=400)
    
    class_name = None
    if image_file:
        image_path = 'uploads/' + image_file.filename
        image_file.save(image_path)

        img = Image.open(image_path).convert("RGB")
        x = test_transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            outputs = cls_model(x)
            _, pred = torch.max(outputs, 1)
            print("Predicted class:", idx_to_class[pred.item()])

            class_name = idx_to_class[pred.item()]
            
    def stream(prompt, class_name=None):
        if class_name:
            prompt = f'{class_name} là gì'
        for token in chat(prompt):
            yield token

    return Response(stream(prompt, class_name), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
