import json   
import torch
from torchvision import transforms

from src.model.cls_model.model import ConvNeXtSwinFusion as cls_model
from config import config

cf = config.get('cls_model_settings', {})
CONFIG = {
    "img size": cf.get('image_size', 224),
    "n classes": cf.get('n_classes', 206),
    "model save path": cf.get('model_save_path', 'cls_model.pth'),
    "class mapping path": cf.get('map_file', 'class_mapping.json'),
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_classification_model():
    model = cls_model(n_classes=CONFIG['num epochs']).to(CONFIG['device'])
    
    checkpoint = torch.load(CONFIG['model save path'], map_location=CONFIG['device'])
    state_dict = checkpoint["model_state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img size'], CONFIG['img size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    try:
        with open(CONFIG['class mapping path'], 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file mapping.")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return cls_model, test_transform, idx_to_class