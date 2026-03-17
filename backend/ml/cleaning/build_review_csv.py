import csv
from pathlib import Path


DEFAULT_REVIEW_FIELDS = [
    "class_name",
    "image_path",
    "flag_count",
    "flags",
    "width",
    "height",
    "aspect_ratio",
    "sharpness_score",
    "grayscale_std",
    "brightness_mean",
    "review_decision",
    "review_reason",
    "review_notes",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"
    suspicious_csv = cleaning_dir / "suspicious_images.csv"
    review_csv = cleaning_dir / "review_candidates.csv"

    if not suspicious_csv.exists():
        raise FileNotFoundError(
            f"Missing suspicious_images.csv. Run find_suspicious_images.py first: {suspicious_csv}"
        )

    rows = []
    with suspicious_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review_row = {field: row.get(field, "") for field in DEFAULT_REVIEW_FIELDS}
            review_row["review_decision"] = ""
            review_row["review_reason"] = ""
            review_row["review_notes"] = ""
            rows.append(review_row)

    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved review CSV to: {review_csv}")
    print(f"Total review candidates: {len(rows)}")
    print("\nAllowed review_decision values you can use:")
    print("  keep")
    print("  remove")
    print("  uncertain")
    print("\nSuggested review_reason values:")
    print("  blurry")
    print("  too_dark")
    print("  low_contrast")
    print("  scene_heavy")
    print("  people_dominant")
    print("  food_too_small")
    print("  wrong_label")
    print("  duplicate_like")
    print("  other")


if __name__ == "__main__":
    main()
