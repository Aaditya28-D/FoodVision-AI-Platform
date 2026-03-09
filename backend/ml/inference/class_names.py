from pathlib import Path
from typing import List


def load_class_names(labels_path: str | Path) -> List[str]:
    labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    class_names = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not class_names:
        raise ValueError("Labels file is empty")

    return class_names