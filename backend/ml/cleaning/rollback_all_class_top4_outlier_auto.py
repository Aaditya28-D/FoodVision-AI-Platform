import csv
from pathlib import Path


TARGET_REASON = "all_class_top4_outlier_auto"


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    review_path = project_root / "backend" / "artifacts" / "dataset_cleaning" / "review_candidates.csv"

    if not review_path.exists():
        raise FileNotFoundError(f"Missing review CSV: {review_path}")

    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    rolled_back = 0

    for row in rows:
        reason = (row.get("review_reason") or "").strip()
        decision = (row.get("review_decision") or "").strip().lower()

        if reason == TARGET_REASON and decision == "remove":
            row["review_decision"] = ""
            row["review_reason"] = ""

            notes = (row.get("review_notes") or "").strip()
            if notes:
                parts = [p.strip() for p in notes.split("|")]
                parts = [p for p in parts if p and p != "all-class-auto-marked"]
                row["review_notes"] = " | ".join(parts)

            rolled_back += 1

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Rollback complete.")
    print(f"Cleared remove decisions for reason={TARGET_REASON}: {rolled_back}")
    print(f"Updated: {review_path}")


if __name__ == "__main__":
    main()
