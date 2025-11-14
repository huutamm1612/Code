from config import config
from PIL import Image
import os
import torch
from torch.utils.data import Dataset

cf = config.get('cls_model_settings', {})

CONFIG = {
    "image size": cf.get('image_size', 224),       
}
class MedicalDataset(Dataset):
    def __init__(self, root_dir, class_mapping, transform=None):
        self.root_dir = root_dir
        self.class_mapping = class_mapping
        self.transform = transform
        self.samples = [] 
        
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_dir):
                continue

            if class_name not in self.class_mapping:
                print(f"Lớp '{class_name}' có trong thư mục ")
                continue
                
            label_index = self.class_mapping[class_name]
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(supported_extensions):
                    image_path = os.path.join(class_dir, file_name)
                    self.samples.append((image_path, label_index))
                    
        print(f"{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Lỗi khi đọc ảnh: {image_path}. Lỗi: {e}")
            return torch.randn(3, CONFIG['image size'], CONFIG['image size']), -1 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label