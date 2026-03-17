import csv
from pathlib import Path


BASE_FIELDS = [
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


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    review_csv = cleaning_dir / "review_candidates.csv"
    outlier_csv = cleaning_dir / "embedding_outliers.csv"
    merged_csv = cleaning_dir / "review_candidates_merged.csv"

    if not review_csv.exists():
        raise FileNotFoundError(f"Missing review CSV: {review_csv}")
    if not outlier_csv.exists():
        raise FileNotFoundError(f"Missing outlier CSV: {outlier_csv}")

    review_rows = read_csv_rows(review_csv)
    outlier_rows = read_csv_rows(outlier_csv)

    merged_map: dict[str, dict] = {}

    # keep all existing review rows first
    for row in review_rows:
        image_path = row["image_path"]
        merged_map[image_path] = {field: row.get(field, "") for field in BASE_FIELDS}

    existing_count = len(merged_map)
    added_from_outliers = 0

    for row in outlier_rows:
        image_path = row["image_path"]
        class_name = row["class_name"]
        outlier_rank = row["rank_within_class"]
        distance = row["distance_from_centroid"]

        if image_path in merged_map:
            # enrich notes for existing row if not already present
            notes = merged_map[image_path].get("review_notes", "") or ""
            extra = f"embedding_outlier rank={outlier_rank} dist={distance}"
            if extra not in notes:
                merged_map[image_path]["review_notes"] = (notes + " | " + extra).strip(" |")
            continue

        merged_map[image_path] = {
            "class_name": class_name,
            "image_path": image_path,
            "flag_count": "0",
            "flags": "",
            "width": "",
            "height": "",
            "aspect_ratio": "",
            "sharpness_score": "",
            "grayscale_std": "",
            "brightness_mean": "",
            "review_decision": "",
            "review_reason": "",
            "review_notes": f"embedding_outlier rank={outlier_rank} dist={distance}",
        }
        added_from_outliers += 1

    merged_rows = sorted(
        merged_map.values(),
        key=lambda x: (x["class_name"], x["image_path"]),
    )

    with merged_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print("Merge complete.\n")
    print(f"Existing review candidates kept: {existing_count}")
    print(f"New outlier-only candidates added: {added_from_outliers}")
    print(f"Total merged review rows: {len(merged_rows)}")
    print(f"Saved merged file: {merged_csv}")


if __name__ == "__main__":
    main()
