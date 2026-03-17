from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA = ROOT / "data"
MODELS = BACKEND / "models"
WEIGHTS = MODELS / "weights"

required_paths = [
    DATA / "food-101" / "images",
    DATA / "food-101" / "meta",
    DATA / "metadata" / "classes.txt",
    DATA / "embeddings" / "food101_resnet50_index.npz",
    WEIGHTS / "efficientnet_b0_best.pth",
    WEIGHTS / "resnet50_best.pth",
    WEIGHTS / "mobilenet_v3_large_best.pth",
    BACKEND / "app" / "main.py",
    ROOT / "web" / "package.json",
]

missing = [str(path) for path in required_paths if not path.exists()]

if missing:
    print("Asset verification failed.\nMissing required files/folders:")
    for item in missing:
        print(f"- {item}")
    sys.exit(1)

print("Asset verification passed.")
