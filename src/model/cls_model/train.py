from src.model.cls_model.model import ConvNeXtSwinFusion as cls_model
from src.model.cls_model.dataset import MedicalDataset as dataset
from config import config

import matplotlib.pyplot as plt
import numpy as np
import json   
import pickle 
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV2

cf = config.get('cls_model_settings', {})

CONFIG = {
    "img size": cf.get('image_size', 224),
    "n classes": cf.get('n_classes', 206),
    "batch size": cf.get('batch_size', 32),
    "num epochs": cf.get('num_epochs', 25),
    "learning rate": cf.get('learning_rate', 0.001),
    "weight decay": cf.get('weight_decay', 1e-4),
    "model save path": cf.get('model_save_path', 'cls_model.pth'),
    "data dir": cf.get('data_dir', 'data/train'),
    "class mapping path": cf.get('map_file', 'class_mapping.json'),
    "seed": cf.get('seed', 42),
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
}
torch.manual_seed(CONFIG['seed'])

def data_loader():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(CONFIG['img size'], scale=(0.7, 1.0)), 
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG['img size'], CONFIG['img size'])),
        transforms.CenterCrop((CONFIG['img size'], CONFIG['img size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        full_dataset = dataset(
            root_dir=CONFIG['data dir'],
            class_mapping=class_to_idx,
            transform=None
        )

        if len(full_dataset) == 0:
            print("Không tìm thấy ảnh nào. "
                "Hãy kiểm tra lại đường dẫn data_dir và file mapping.")
        else:
            total_size = len(full_dataset)
            val_size = int(total_size * 0.2)
            test_size = int(total_size * 0.1) + 1
            train_size = total_size - val_size - test_size

            print(f"\nTổng số ảnh: {total_size}")
            print(f"Kích thước tập Train: {train_size} ảnh")
            print(f"Kích thước tập Validation:  {val_size} ảnh")
            print(f"Kích thước tập Test:  {test_size} ảnh")

            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, 
                [train_size, val_size, test_size]
            )
            
            train_dataset.dataset.transform = train_transforms
            val_dataset.dataset.transform = val_transforms
            test_dataset.dataset.transform = val_transforms
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch size'], shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch size'], shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch size'], shuffle=False, pin_memory=True)
            
            return train_loader, val_loader, test_loader
            
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy đường dẫn dữ liệu '{CONFIG['data dir']}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    try:
        with open(CONFIG['class mapping path'], 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
            print(f"Đã tải thành công file mapping: {CONFIG['class mapping path']}")
    except FileNotFoundError:
        print(f"Không tìm thấy file mapping '{CONFIG['class mapping path']}'.")
        
    train_loader, val_loader, test_loader = data_loader()
    
    model = cls_model(n_classes=CONFIG['n classes']).to(CONFIG['device'])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    mixup_fn = Mixup(
        mixup_alpha=0.4,  
        cutmix_alpha=0.4,  
        prob=0.7,          
        switch_prob=0.5,   
        mode='batch',     
        label_smoothing=0.1, 
        num_classes=CONFIG['n classes']
    )
    
    criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()
    optimizer = create_optimizer_v2(model, opt='adamw', lr=CONFIG['learning rate'], weight_decay=CONFIG['weight decay'])
    scheduler = CosineLRScheduler(optimizer, t_initial=CONFIG['num epochs'], lr_min=5e-6)
    scaler = torch.amp.GradScaler(CONFIG['device'])
    model_ema = ModelEmaV2(model)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    initial_decay = 0.995
    final_decay = 0.9998
    
    for epoch in range(1, CONFIG['num epochs']+1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        decay = final_decay - (final_decay - initial_decay) * (1 - epoch / CONFIG['num epochs'])**2
        model_ema.decay = decay

        for imgs, labels in loop:
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])

            if mixup_fn is not None:
                imgs, labels = mixup_fn(imgs, labels)

            optimizer.zero_grad()
            with torch.amp.autocast(CONFIG['device']):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model_ema.update(model)

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels.argmax(dim=1)).sum().item() if mixup_fn else (preds == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=f"{train_correct/train_total:.4f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for imgs, labels in loop:
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                with torch.amp.autocast(CONFIG['device']):
                    outputs = model(imgs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                loop.set_postfix(loss=loss.item())

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG['model save path'])
            print("Model improved and saved!")
            
            
    epochs = range(len(train_losses))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'o-', color='tab:green', label='Train Accuracy', linewidth=2, markersize=6)
    plt.plot(epochs, val_accuracies,   's-', color='tab:blue',  label='Val Accuracy',   linewidth=2, markersize=6)

    plt.title('Train vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'o-', color='tab:red',    label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses,   's-', color='tab:orange', label='Val Loss',   linewidth=2, markersize=6)

    plt.title('Train vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()
    
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Epoch {epoch} [Val]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            test_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            loop.set_postfix(loss=loss.item())
    test_loss /= test_total
    test_acc = test_correct / test_total

    print(f"| Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f}")