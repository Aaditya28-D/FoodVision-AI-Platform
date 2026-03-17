from pathlib import Path

from ml.retrieval.indexer import RetrievalIndexer


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    dataset_root = project_root / "data" / "food-101" / "images"
    output_path = project_root / "data" / "embeddings" / "food101_resnet50_index.npz"
    keep_manifest_path = (
        project_root
        / "backend"
        / "artifacts"
        / "dataset_cleaning"
        / "cleaned_keep_manifest.txt"
    )

    indexer = RetrievalIndexer(
        dataset_root=dataset_root,
        output_path=output_path,
        keep_manifest_path=keep_manifest_path,
        device="auto",
    )
    indexer.build()


if __name__ == "__main__":
    main()
