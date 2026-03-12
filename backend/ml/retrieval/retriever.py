from pathlib import Path

import numpy as np
from PIL import Image

from ml.retrieval.embedder import ImageEmbedder


class SimilarDishRetriever:
    def __init__(
        self,
        index_path: str | Path,
        dataset_root: str | Path,
        device: str = "auto",
    ) -> None:
        self.index_path = Path(index_path)
        self.dataset_root = Path(dataset_root)
        self.embedder = ImageEmbedder(device=device)

        if not self.index_path.exists():
            raise FileNotFoundError(f"Retrieval index not found: {self.index_path}")

        data = np.load(self.index_path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.image_paths = data["image_paths"]
        self.class_names = data["class_names"]

    def retrieve(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        query = self.embedder.embed_pil(image).numpy()
        similarities = self.embeddings @ query

        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            rel_path = str(self.image_paths[idx])
            results.append(
                {
                    "rank": rank,
                    "class_name": str(self.class_names[idx]),
                    "image_path": rel_path,
                    "similarity": float(similarities[idx]),
                }
            )

        return results