from pathlib import Path


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_rel_path(path_str: str) -> str:
    path_str = path_str.strip()
    if path_str.endswith(".jpg"):
        path_str = path_str[:-4]
    return path_str


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    meta_dir = project_root / "data" / "food-101" / "meta"
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    train_split_path = meta_dir / "train_split.txt"
    val_split_path = meta_dir / "val_split.txt"
    remove_manifest_path = cleaning_dir / "cleaned_remove_manifest.txt"

    train_split_cleaned_path = meta_dir / "train_split_cleaned.txt"
    val_split_cleaned_path = meta_dir / "val_split_cleaned.txt"
    summary_path = cleaning_dir / "cleaned_train_val_split_summary.txt"

    train_items = read_lines(train_split_path)
    val_items = read_lines(val_split_path)

    remove_items = set()
    if remove_manifest_path.exists():
        remove_items = {normalize_rel_path(x) for x in read_lines(remove_manifest_path)}

    cleaned_train_items = [
        item for item in train_items
        if normalize_rel_path(item) not in remove_items
    ]
    cleaned_val_items = [
        item for item in val_items
        if normalize_rel_path(item) not in remove_items
    ]

    removed_from_train = len(train_items) - len(cleaned_train_items)
    removed_from_val = len(val_items) - len(cleaned_val_items)

    write_lines(train_split_cleaned_path, cleaned_train_items)
    write_lines(val_split_cleaned_path, cleaned_val_items)

    summary_lines = [
        f"original_train_split_count={len(train_items)}",
        f"cleaned_train_split_count={len(cleaned_train_items)}",
        f"removed_from_train_split={removed_from_train}",
        f"original_val_split_count={len(val_items)}",
        f"cleaned_val_split_count={len(cleaned_val_items)}",
        f"removed_from_val_split={removed_from_val}",
        f"total_removed_manifest={len(remove_items)}",
    ]
    write_lines(summary_path, summary_lines)

    print("Cleaned train/val splits created.\n")
    print(f"Train split cleaned file: {train_split_cleaned_path}")
    print(f"Val split cleaned file: {val_split_cleaned_path}")
    print(f"Summary file: {summary_path}\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
