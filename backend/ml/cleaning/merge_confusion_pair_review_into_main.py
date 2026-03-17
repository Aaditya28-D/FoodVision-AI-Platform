import csv
from pathlib import Path

MAIN_FIELDS = [
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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    main_review_path = cleaning_dir / "review_candidates.csv"
    confusion_review_path = cleaning_dir / "confusion_pair_review" / "confusion_pair_review_candidates.csv"

    if not main_review_path.exists():
        raise FileNotFoundError(f"Missing main review CSV: {main_review_path}")
    if not confusion_review_path.exists():
        raise FileNotFoundError(f"Missing confusion review CSV: {confusion_review_path}")

    main_rows = read_csv(main_review_path)
    confusion_rows = read_csv(confusion_review_path)

    merged = {}
    for row in main_rows:
        merged[row["image_path"]] = {field: row.get(field, "") for field in MAIN_FIELDS}

    updates = 0
    adds = 0

    for row in confusion_rows:
        image_path = row["image_path"]
        existing = merged.get(image_path)

        if existing is None:
            merged[image_path] = {
                "class_name": row["class_name"],
                "image_path": image_path,
                "flag_count": "0",
                "flags": "",
                "width": "",
                "height": "",
                "aspect_ratio": "",
                "sharpness_score": "",
                "grayscale_std": "",
                "brightness_mean": "",
                "review_decision": row.get("review_decision", ""),
                "review_reason": row.get("review_reason", ""),
                "review_notes": row.get("review_notes", ""),
            }
            adds += 1
            continue

        # only update undecided main rows
        existing_decision = (existing.get("review_decision") or "").strip().lower()
        new_decision = (row.get("review_decision") or "").strip().lower()

        if existing_decision == "" and new_decision in {"keep", "remove", "uncertain"}:
            existing["review_decision"] = row.get("review_decision", "")
            existing["review_reason"] = row.get("review_reason", "")
            note_old = existing.get("review_notes", "") or ""
            note_new = row.get("review_notes", "") or ""
            combined = " | ".join([x for x in [note_old, note_new] if x]).strip(" |")
            existing["review_notes"] = combined
            updates += 1
        else:
            note_old = existing.get("review_notes", "") or ""
            note_new = row.get("review_notes", "") or ""
            if note_new and note_new not in note_old:
                existing["review_notes"] = (note_old + " | " + note_new).strip(" |")

    merged_rows = sorted(merged.values(), key=lambda x: (x["class_name"], x["image_path"]))

    with main_review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print("Merged confusion-pair review into main review CSV.\n")
    print(f"Added new rows: {adds}")
    print(f"Updated undecided rows: {updates}")
    print(f"Main review file updated: {main_review_path}")


if __name__ == "__main__":
    main()
