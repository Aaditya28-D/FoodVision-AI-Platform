import csv
from pathlib import Path


def split_flags(flag_string: str) -> set[str]:
    if not flag_string:
        return set()
    return {item.strip() for item in flag_string.split("|") if item.strip()}


def should_remove(flags: set[str], outlier_rank: int | None, flag_count: int) -> tuple[bool, str]:
    # Rule 1: very blurry is always bad
    if "very_blurry" in flags:
        return True, "very_blurry_auto"

    # Rule 2: dark + blurry/low contrast
    if "too_dark" in flags and ("low_contrast" in flags or "blurry" in flags):
        return True, "dark_low_visibility_auto"

    # Rule 3: tiny and weird aspect ratio
    if "small_image" in flags and "extreme_aspect_ratio" in flags:
        return True, "small_extreme_ratio_auto"

    # Rule 4: suspicious + strong outlier
    if outlier_rank is not None and outlier_rank <= 15:
        if {"blurry", "too_dark", "low_contrast"} & flags:
            return True, "suspicious_outlier_auto"

    # Rule 5: any flagged sample in top-5 outliers
    if outlier_rank is not None and outlier_rank <= 5 and flag_count >= 1:
        return True, "top5_outlier_flagged_auto"

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

            image_path = row["image_path"]
            flags = split_flags(row.get("flags", ""))
            flag_count = int(float(row.get("flag_count", 0) or 0))
            outlier_rank = outlier_rank_map.get(image_path)

            remove_it, reason = should_remove(flags, outlier_rank, flag_count)

            if remove_it:
                row["review_decision"] = "remove"
                row["review_reason"] = reason
                note = row.get("review_notes", "") or ""
                row["review_notes"] = (note + " | auto-marked").strip(" |")
                auto_removed += 1

            rows.append(row)

    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Aggressive auto-review complete.\n")
    print(f"Auto-marked remove rows: {auto_removed}")
    print(f"Already reviewed rows left untouched: {skipped_already_reviewed}")
    print(f"Updated file: {review_csv}")


if __name__ == "__main__":
    main()
