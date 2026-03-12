from pathlib import Path

import numpy as np

from ml.retrieval.embedder import ImageEmbedder


class RetrievalIndexer:
    def __init__(self, dataset_root: str | Path, output_path: str | Path, device: str = "auto") -> None:
        self.dataset_root = Path(dataset_root)
        self.output_path = Path(output_path)
        self.embedder = ImageEmbedder(device=device)

    def build(self) -> None:
        image_paths = sorted(self.dataset_root.glob("*/*.jpg"))

        if not image_paths:
            raise FileNotFoundError(f"No dataset images found under: {self.dataset_root}")

        embeddings = []
        relative_paths = []
        class_names = []

        for idx, image_path in enumerate(image_paths, start=1):
            embedding = self.embedder.embed_path(image_path).numpy()
            embeddings.append(embedding)

            rel_path = image_path.relative_to(self.dataset_root)
            relative_paths.append(str(rel_path))
            class_names.append(image_path.parent.name)

            if idx % 1000 == 0:
                print(f"Indexed {idx} images...")

        embeddings_array = np.vstack(embeddings)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output_path,
            embeddings=embeddings_array,
            image_paths=np.array(relative_paths, dtype=object),
            class_names=np.array(class_names, dtype=object),
        )

        print(f"Saved retrieval index to: {self.output_path}")
        print(f"Total indexed images: {len(relative_paths)}")