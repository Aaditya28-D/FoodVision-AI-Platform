import csv
from pathlib import Path

TARGET_CLASSES = {
    "cheese_plate",
    "cheesecake",
    "steak",
    "filet_mignon",
    "prime_rib",
    "pork_chop",
    "beef_tartare",
    "tuna_tartare",
    "chocolate_mousse",
    "chocolate_cake",
    "ravioli",
    "gnocchi",
    "apple_pie",
    "bread_pudding",
}


def should_remove(class_name: str, rank: int) -> tuple[bool, str]:
    if class_name not in TARGET_CLASSES:
        return False, ""

    # strongest candidates only
    if rank <= 6:
        return True, "confusion_pair_top6_outlier_auto"

    return False, ""


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    review_path = (
        project_root
        / "backend"
        / "artifacts"
        / "dataset_cleaning"
        / "confusion_pair_review"
        / "confusion_pair_review_candidates.csv"
    )

    if not review_path.exists():
        raise FileNotFoundError(f"Missing confusion review CSV: {review_path}")

    rows = []
    auto_removed = 0
    untouched_reviewed = 0

    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            decision = (row.get("review_decision") or "").strip().lower()
            if decision in {"keep", "remove", "uncertain"}:
                untouched_reviewed += 1
                rows.append(row)
                continue

            class_name = row["class_name"]
            rank = int(row["rank_within_class"])

            remove_it, reason = should_remove(class_name, rank)
            if remove_it:
                row["review_decision"] = "remove"
                row["review_reason"] = reason
                note = row.get("review_notes", "") or ""
                row["review_notes"] = (note + " | confusion-auto-marked").strip(" |")
                auto_removed += 1

            rows.append(row)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Confusion-pair auto-mark complete.\n")
    print(f"Newly auto-marked remove rows: {auto_removed}")
    print(f"Already reviewed rows left untouched: {untouched_reviewed}")
    print(f"Updated file: {review_path}")


if __name__ == "__main__":
    main()
