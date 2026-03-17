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

    def retrieve(
        self,
        image: Image.Image,
        top_k: int = 8,
        predicted_class: str | None = None,
        max_other_per_class: int = 2,
        exact_match_threshold: float = 0.9999,
    ) -> dict:
        query = self.embedder.embed_pil(image).numpy()
        similarities = self.embeddings @ query

        sorted_indices = np.argsort(-similarities)

        exact_match_found = False
        exact_match_item: dict | None = None
        same_class_results: list[dict] = []
        other_results: list[dict] = []

        same_rank = 1
        other_rank = 1
        other_class_counts: dict[str, int] = {}

        for idx in sorted_indices:
            similarity = float(similarities[idx])
            rel_path = str(self.image_paths[idx])
            class_name = str(self.class_names[idx])

            item = {
                "class_name": class_name,
                "image_path": rel_path,
                "similarity": similarity,
            }

            if similarity >= exact_match_threshold:
                exact_match_found = True
                if exact_match_item is None:
                    exact_match_item = item
                continue

            if predicted_class is not None and class_name == predicted_class:
                if len(same_class_results) < top_k:
                    same_class_results.append(
                        {
                            "rank": same_rank,
                            **item,
                        }
                    )
                    same_rank += 1
            else:
                current_count = other_class_counts.get(class_name, 0)

                if current_count < max_other_per_class and len(other_results) < top_k:
                    other_results.append(
                        {
                            "rank": other_rank,
                            **item,
                        }
                    )
                    other_rank += 1
                    other_class_counts[class_name] = current_count + 1

            if len(same_class_results) >= top_k and len(other_results) >= top_k:
                break

        return {
            "predicted_class": predicted_class,
            "exact_match_found": exact_match_found,
            "exact_match_item": exact_match_item,
            "same_class_results": same_class_results,
            "other_results": other_results,
        }
