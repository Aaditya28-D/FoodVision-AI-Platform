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
    audit_path = cleaning_dir / "all_class_audit" / "all_class_audit_candidates.csv"

    if not main_review_path.exists():
        raise FileNotFoundError(f"Missing main review CSV: {main_review_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing all-class audit CSV: {audit_path}")

    main_rows = read_csv(main_review_path)
    audit_rows = read_csv(audit_path)

    merged = {}
    for row in main_rows:
        merged[row["image_path"]] = {field: row.get(field, "") for field in MAIN_FIELDS}

    adds = 0
    updates = 0

    for row in audit_rows:
        image_path = row["image_path"]
        audit_decision = (row.get("audit_decision") or "").strip().lower()
        audit_reason = row.get("audit_reason", "")
        audit_notes = row.get("audit_notes", "") or ""

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
                "review_decision": audit_decision,
                "review_reason": audit_reason,
                "review_notes": audit_notes,
            }
            adds += 1
            continue

        existing_decision = (existing.get("review_decision") or "").strip().lower()

        if existing_decision == "" and audit_decision in {"keep", "remove", "uncertain"}:
            existing["review_decision"] = audit_decision
            existing["review_reason"] = audit_reason
            old_notes = existing.get("review_notes", "") or ""
            existing["review_notes"] = " | ".join([x for x in [old_notes, audit_notes] if x]).strip(" |")
            updates += 1
        else:
            old_notes = existing.get("review_notes", "") or ""
            if audit_notes and audit_notes not in old_notes:
                existing["review_notes"] = (old_notes + " | " + audit_notes).strip(" |")

    merged_rows = sorted(merged.values(), key=lambda x: (x["class_name"], x["image_path"]))

    with main_review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print("Merged all-class audit into main review CSV.\n")
    print(f"Added new rows: {adds}")
    print(f"Updated undecided rows: {updates}")
    print(f"Main review file updated: {main_review_path}")


if __name__ == "__main__":
    main()
