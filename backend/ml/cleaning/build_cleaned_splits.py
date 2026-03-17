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

    metadata_dir = project_root / "data" / "metadata"
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    train_txt = metadata_dir / "train.txt"
    test_txt = metadata_dir / "test.txt"
    remove_manifest = cleaning_dir / "cleaned_remove_manifest.txt"

    train_cleaned_txt = metadata_dir / "train_cleaned.txt"
    test_cleaned_txt = metadata_dir / "test_cleaned.txt"
    split_summary_txt = cleaning_dir / "cleaned_split_summary.txt"

    train_items = read_lines(train_txt)
    test_items = read_lines(test_txt)

    remove_items = set()
    if remove_manifest.exists():
        remove_items = {normalize_rel_path(x) for x in read_lines(remove_manifest)}

    train_cleaned = [item for item in train_items if normalize_rel_path(item) not in remove_items]
    test_cleaned = [item for item in test_items if normalize_rel_path(item) not in remove_items]

    removed_from_train = len(train_items) - len(train_cleaned)
    removed_from_test = len(test_items) - len(test_cleaned)

    write_lines(train_cleaned_txt, train_cleaned)
    write_lines(test_cleaned_txt, test_cleaned)

    summary_lines = [
        f"original_train_count={len(train_items)}",
        f"cleaned_train_count={len(train_cleaned)}",
        f"removed_from_train={removed_from_train}",
        f"original_test_count={len(test_items)}",
        f"cleaned_test_count={len(test_cleaned)}",
        f"removed_from_test={removed_from_test}",
        f"total_removed_manifest={len(remove_items)}",
    ]
    write_lines(split_summary_txt, summary_lines)

    print("Cleaned splits created.\n")
    print(f"Train cleaned file: {train_cleaned_txt}")
    print(f"Test cleaned file: {test_cleaned_txt}")
    print(f"Summary file: {split_summary_txt}\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
