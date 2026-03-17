import csv
import shutil
from pathlib import Path


def read_nonempty_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "data" / "food-101" / "images"
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    remove_list_path = cleaning_dir / "remove_list.txt"
    keep_list_path = cleaning_dir / "keep_list.txt"
    uncertain_list_path = cleaning_dir / "uncertain_list.txt"

    quarantine_root = cleaning_dir / "quarantine_removed"
    cleaned_keep_manifest = cleaning_dir / "cleaned_keep_manifest.txt"
    cleaned_remove_manifest = cleaning_dir / "cleaned_remove_manifest.txt"
    summary_csv = cleaning_dir / "cleaned_dataset_summary.csv"

    remove_paths = read_nonempty_lines(remove_list_path)
    keep_paths = read_nonempty_lines(keep_list_path)
    uncertain_paths = read_nonempty_lines(uncertain_list_path)

    all_dataset_files = sorted(
        str(path.relative_to(dataset_root))
        for path in dataset_root.glob("*/*.jpg")
    )

    if not all_dataset_files:
        raise FileNotFoundError(f"No dataset images found under: {dataset_root}")

    quarantine_root.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_remove_paths = []

    for rel_path_str in sorted(remove_paths):
        src = dataset_root / rel_path_str
        dst = quarantine_root / rel_path_str

        if not src.exists():
            missing_remove_paths.append(rel_path_str)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_count += 1

    cleaned_keep_paths = []
    cleaned_remove_paths = sorted(remove_paths)

    remove_set = set(remove_paths)

    for rel_path_str in all_dataset_files:
        if rel_path_str not in remove_set:
            cleaned_keep_paths.append(rel_path_str)

    cleaned_keep_manifest.write_text(
        "\n".join(cleaned_keep_paths) + ("\n" if cleaned_keep_paths else ""),
        encoding="utf-8",
    )

    cleaned_remove_manifest.write_text(
        "\n".join(cleaned_remove_paths) + ("\n" if cleaned_remove_paths else ""),
        encoding="utf-8",
    )

    summary_rows = [
        {
            "metric": "total_dataset_images",
            "value": len(all_dataset_files),
        },
        {
            "metric": "review_keep_count",
            "value": len(keep_paths),
        },
        {
            "metric": "review_remove_count",
            "value": len(remove_paths),
        },
        {
            "metric": "review_uncertain_count",
            "value": len(uncertain_paths),
        },
        {
            "metric": "quarantine_copied_count",
            "value": copied_count,
        },
        {
            "metric": "quarantine_missing_remove_paths",
            "value": len(missing_remove_paths),
        },
        {
            "metric": "cleaned_keep_manifest_count",
            "value": len(cleaned_keep_paths),
        },
        {
            "metric": "cleaned_remove_manifest_count",
            "value": len(cleaned_remove_paths),
        },
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Phase 6 complete.\n")
    print(f"Dataset root: {dataset_root}")
    print(f"Quarantine folder: {quarantine_root}")
    print(f"Copied removed images: {copied_count}")
    print(f"Missing remove paths: {len(missing_remove_paths)}")
    print(f"Cleaned keep manifest: {cleaned_keep_manifest}")
    print(f"Cleaned remove manifest: {cleaned_remove_manifest}")
    print(f"Summary CSV: {summary_csv}")

    if missing_remove_paths:
        print("\nFirst missing remove paths:")
        for item in missing_remove_paths[:20]:
            print(item)


if __name__ == "__main__":
    main()
