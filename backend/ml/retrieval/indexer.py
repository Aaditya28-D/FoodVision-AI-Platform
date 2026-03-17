from pathlib import Path

import numpy as np

from ml.retrieval.embedder import ImageEmbedder


class RetrievalIndexer:
    def __init__(
        self,
        dataset_root: str | Path,
        output_path: str | Path,
        device: str = "auto",
        keep_manifest_path: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.output_path = Path(output_path)
        self.keep_manifest_path = Path(keep_manifest_path) if keep_manifest_path else None
        self.embedder = ImageEmbedder(device=device)

    def _load_image_paths(self) -> list[Path]:
        if self.keep_manifest_path is not None:
            if not self.keep_manifest_path.exists():
                raise FileNotFoundError(f"Keep manifest not found: {self.keep_manifest_path}")

            image_paths = []
            with self.keep_manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue

                    full_path = self.dataset_root / rel_path
                    if full_path.exists():
                        image_paths.append(full_path)

            return sorted(image_paths)

        return sorted(self.dataset_root.glob("*/*.jpg"))

    def build(self) -> None:
        image_paths = self._load_image_paths()

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
