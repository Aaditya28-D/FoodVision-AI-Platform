import csv
from pathlib import Path

TARGET_CLASSES = {
    "pizza",
    "tacos",
    "hamburger",
    "hot_dog",
    "steak",
    "ramen",
    "sushi",
}


def split_flags(flag_string: str) -> set[str]:
    if not flag_string:
        return set()
    return {item.strip() for item in flag_string.split("|") if item.strip()}


def should_remove(class_name: str, outlier_rank: int | None, flags: set[str], flag_count: int) -> tuple[bool, str]:
    if class_name not in TARGET_CLASSES:
        return False, ""

    if outlier_rank is None:
        return False, ""

    # strongest rule: top 8 outliers in target classes
    if outlier_rank <= 8:
        return True, "focused_top8_outlier_auto"

    # still strong: top 15 outlier + any suspicious flag
    if outlier_rank <= 15 and flag_count >= 1:
        return True, "focused_outlier_flagged_auto"

    # medium outlier + stronger technical issue
    if outlier_rank <= 20 and (
        "very_blurry" in flags
        or ("too_dark" in flags and "low_contrast" in flags)
        or ("small_image" in flags and "extreme_aspect_ratio" in flags)
    ):
        return True, "focused_outlier_quality_auto"

    return False, ""


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    review_csv = cleaning_dir / "review_candidates.csv"
    outlier_csv = cleaning_dir / "embedding_outliers.csv"

    if not review_csv.exists():
        raise FileNotFoundError(f"Missing review CSV: {review_csv}")
    if not outlier_csv.exists():
        raise FileNotFoundError(f"Missing outlier CSV: {outlier_csv}")

    outlier_rank_map = {}
    with outlier_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"]
            rank = int(row["rank_within_class"])
            current = outlier_rank_map.get(image_path)
            if current is None or rank < current:
                outlier_rank_map[image_path] = rank

    rows = []
    auto_removed = 0
    skipped_already_reviewed = 0

    with review_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            decision = (row.get("review_decision") or "").strip().lower()
            if decision in {"keep", "remove", "uncertain"}:
                skipped_already_reviewed += 1
                rows.append(row)
                continue

            class_name = row["class_name"]
            image_path = row["image_path"]
            flags = split_flags(row.get("flags", ""))
            flag_count = int(float(row.get("flag_count", 0) or 0))
            outlier_rank = outlier_rank_map.get(image_path)

            remove_it, reason = should_remove(class_name, outlier_rank, flags, flag_count)

            if remove_it:
                row["review_decision"] = "remove"
                row["review_reason"] = reason
                note = row.get("review_notes", "") or ""
                row["review_notes"] = (note + " | focused-auto-marked").strip(" |")
                auto_removed += 1

            rows.append(row)

    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Focused deep clean complete.\n")
    print(f"Target classes: {', '.join(sorted(TARGET_CLASSES))}")
    print(f"Newly auto-marked remove rows: {auto_removed}")
    print(f"Already reviewed rows left untouched: {skipped_already_reviewed}")
    print(f"Updated file: {review_csv}")


if __name__ == "__main__":
    main()
