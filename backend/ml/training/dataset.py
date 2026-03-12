from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        self._build_samples()

    def _build_samples(self) -> None:
        class_names = sorted([path.name for path in self.data_dir.iterdir() if path.is_dir()])

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        with self.split_file.open("r", encoding="utf-8") as file:
            for line in file:
                relative_no_ext = line.strip()
                if not relative_no_ext:
                    continue

                class_name = relative_no_ext.split("/")[0]
                image_path = self.data_dir / f"{relative_no_ext}.jpg"

                if not image_path.exists():
                    continue

                class_index = self.class_to_idx[class_name]
                self.samples.append((image_path, class_index))

        if not self.samples:
            raise RuntimeError(
                f"No valid samples were found using split file {self.split_file} and data dir {self.data_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_index = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_index