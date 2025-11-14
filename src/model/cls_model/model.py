from config import config
from PIL import Image
import os
import json   
import pickle 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchsummary import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import timm
from timm.data.mixup import Mixup

cf = config.get('cls_model_settings', {})

CONFIG = {
    "convnext model": cf.get('convnext_model', 'convnext_base'),
    "swin model": cf.get('swin_model', 'swin_base_patch4_window7_224'),
    "n classes": cf.get('n_classes', 206),
}

class AttentionFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, conv_feat, swin_feat):
        fusion_input = torch.cat([conv_feat, swin_feat], dim=1)
        alpha = self.fc(fusion_input) 
        fused = alpha * conv_feat + (1 - alpha) * swin_feat
        return fused

class ConvNeXtSwinFusion(nn.Module):
    def __init__(self, n_classes=206, dropout=0.3, pretrained=True, freeze_backbones=False):
        super().__init__()

        self.convnext = timm.create_model(
            CONFIG['convnext model'], pretrained=pretrained, num_classes=0
        )
        conv_dim = self.convnext.num_features

        self.swin = timm.create_model(
            CONFIG['swin model'], pretrained=pretrained, num_classes=0
        )
        swin_dim = self.swin.num_features

        assert conv_dim == swin_dim, "Hai backbone phải có cùng số chiều đặc trưng!"

        self.fusion = AttentionFusion(conv_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(conv_dim),
            nn.Linear(conv_dim, 1536),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1536, n_classes)
        )

        if freeze_backbones:
            for p in self.convnext.parameters():
                p.requires_grad = False
            for p in self.swin.parameters():
                p.requires_grad = False

    def forward(self, x):
        conv_feat = self.convnext(x)  # (B, 768)
        swin_feat = self.swin(x)      # (B, 768)

        fused = self.fusion(conv_feat, swin_feat)
        out = self.classifier(fused)

        return out