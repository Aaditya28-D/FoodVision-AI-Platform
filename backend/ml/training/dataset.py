from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class Food101DatasetFromSplit(Dataset):
    def __init__(
        self,
        images_root: str | Path,
        split_file: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.split_file = Path(split_file)
        self.transform = transform

        if not self.images_root.exists():
            raise FileNotFoundError(f"Images root not found: {self.images_root}")

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        lines = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        class_to_index: dict[str, int] = {}

        for item in lines:
            class_name, image_id = item.rsplit("/", 1)

            if class_name not in class_to_index:
                class_to_index[class_name] = len(class_to_index)

            image_path = self.images_root / f"{item}.jpg"
            label_index = class_to_index[class_name]
            samples.append((image_path, label_index))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label