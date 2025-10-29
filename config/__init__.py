import yaml
from pathlib import Path

config_dir = Path(__file__).parent

config_path = config_dir / "model_config.yaml"

config = {}

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file config tại: {config_path}")
except Exception as e:
    print(f"LỖI: Không thể đọc file YAML: {e}")