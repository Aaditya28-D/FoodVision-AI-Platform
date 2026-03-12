from pathlib import Path
import random


def main() -> None:
    random.seed(42)

    project_root = Path(__file__).resolve().parents[3]
    meta_dir = project_root / "data" / "food-101" / "meta"

    original_train_path = meta_dir / "train.txt"
    train_split_path = meta_dir / "train_split.txt"
    val_split_path = meta_dir / "val_split.txt"

    if not original_train_path.exists():
        raise FileNotFoundError(f"Missing file: {original_train_path}")

    class_to_items: dict[str, list[str]] = {}

    with original_train_path.open("r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if not item:
                continue

            class_name = item.split("/")[0]
            class_to_items.setdefault(class_name, []).append(item)

    train_items: list[str] = []
    val_items: list[str] = []

    for class_name, items in sorted(class_to_items.items()):
        items = items[:]
        random.shuffle(items)

        # Food-101 train split has 750 images per class
        # Use 100 per class for validation, 650 for training
        val_count = 100
        val_part = items[:val_count]
        train_part = items[val_count:]

        val_items.extend(val_part)
        train_items.extend(train_part)

    with train_split_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(train_items) + "\n")

    with val_split_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(val_items) + "\n")

    print(f"Created: {train_split_path}")
    print(f"Created: {val_split_path}")
    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")


if __name__ == "__main__":
    main()