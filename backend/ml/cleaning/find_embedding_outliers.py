import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    index_path = project_root / "data" / "embeddings" / "food101_resnet50_index.npz"
    output_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"
    output_dir.mkdir(parents=True, exist_ok=True)

    outlier_csv = output_dir / "embedding_outliers.csv"
    summary_csv = output_dir / "embedding_outlier_summary_by_class.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {index_path}")

    data = np.load(index_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]
    class_names = data["class_names"]

    class_to_indices = defaultdict(list)
    for idx, class_name in enumerate(class_names):
        class_to_indices[str(class_name)].append(idx)

    outlier_rows = []
    summary_rows = []

    for class_name, indices in sorted(class_to_indices.items()):
        class_embeddings = embeddings[indices]
        if len(class_embeddings) < 2:
            continue

        centroid = class_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        distances = []
        for local_idx, global_idx in enumerate(indices):
            emb = class_embeddings[local_idx]
            dist = float(np.linalg.norm(emb - centroid))
            distances.append((global_idx, dist))

        distances.sort(key=lambda x: x[1], reverse=True)

        top_n = min(25, len(distances))
        top_rows = distances[:top_n]

        distance_values = [d for _, d in distances]
        summary_rows.append(
            {
                "class_name": class_name,
                "num_images": len(indices),
                "max_distance": round(max(distance_values), 6),
                "mean_distance": round(float(np.mean(distance_values)), 6),
                "top_5_mean_distance": round(float(np.mean([d for _, d in distances[:5]])), 6),
            }
        )

        for rank, (global_idx, distance) in enumerate(top_rows, start=1):
            outlier_rows.append(
                {
                    "class_name": class_name,
                    "rank_within_class": rank,
                    "image_path": str(image_paths[global_idx]),
                    "distance_from_centroid": round(distance, 6),
                }
            )

    outlier_rows.sort(key=lambda x: (x["class_name"], x["rank_within_class"]))

    with outlier_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "rank_within_class",
                "image_path",
                "distance_from_centroid",
            ],
        )
        writer.writeheader()
        writer.writerows(outlier_rows)

    summary_rows.sort(key=lambda x: x["top_5_mean_distance"], reverse=True)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "num_images",
                "max_distance",
                "mean_distance",
                "top_5_mean_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Done.\n")
    print(f"Saved outlier rows to: {outlier_csv}")
    print(f"Saved summary by class to: {summary_csv}")
    print(f"Total outlier candidates saved: {len(outlier_rows)}")


if __name__ == "__main__":
    main()
