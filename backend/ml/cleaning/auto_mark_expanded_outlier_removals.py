import csv
from pathlib import Path

TARGET_CLASSES = {
    "apple_pie",
    "pho",
    "peking_duck",
    "beignets",
    "lasagna",
    "chicken_curry",
    "fried_rice",
    "donuts",
    "chicken_wings",
    "tacos",
    "hot_dog",
    "hamburger",
}


def should_remove(class_name: str, rank: int) -> tuple[bool, str]:
    if class_name not in TARGET_CLASSES:
        return False, ""

    if rank <= 6:
        return True, "expanded_outlier_top6_auto"

    return False, ""


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    review_path = (
        project_root
        / "backend"
        / "artifacts"
        / "dataset_cleaning"
        / "expanded_outlier_review"
        / "expanded_outlier_review_candidates.csv"
    )

    if not review_path.exists():
        raise FileNotFoundError(f"Missing expanded outlier review CSV: {review_path}")

    rows = []
    auto_removed = 0

    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            decision = (row.get("review_decision") or "").strip().lower()
            if decision in {"keep", "remove", "uncertain"}:
                rows.append(row)
                continue

            class_name = row["class_name"]
            rank = int(row["rank_within_class"])

            remove_it, reason = should_remove(class_name, rank)
            if remove_it:
                row["review_decision"] = "remove"
                row["review_reason"] = reason
                note = row.get("review_notes", "") or ""
                row["review_notes"] = (note + " | expanded-outlier-auto-marked").strip(" |")
                auto_removed += 1

            rows.append(row)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Expanded outlier auto-mark complete.\n")
    print(f"Newly auto-marked remove rows: {auto_removed}")
    print(f"Updated file: {review_path}")


if __name__ == "__main__":
    main()
