import csv
from pathlib import Path


def parse_rank(rank_info: str) -> int | None:
    rank_info = (rank_info or "").strip()
    if not rank_info.startswith("rank="):
        return None
    try:
        return int(rank_info.split("=", 1)[1])
    except ValueError:
        return None


def should_remove(source: str, rank: int | None) -> tuple[bool, str]:
    # already removed/quarantined examples should remain remove
    if source == "removed":
        return True, "all_class_removed_source_auto"

    # strongest global outliers only
    if source == "outlier" and rank is not None and rank <= 4:
        return True, "all_class_top4_outlier_auto"

    return False, ""


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    review_path = (
        project_root
        / "backend"
        / "artifacts"
        / "dataset_cleaning"
        / "all_class_audit"
        / "all_class_audit_candidates.csv"
    )

    if not review_path.exists():
        raise FileNotFoundError(f"Missing all-class audit CSV: {review_path}")

    rows = []
    auto_removed = 0
    untouched_reviewed = 0

    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            decision = (row.get("audit_decision") or "").strip().lower()
            if decision in {"keep", "remove", "uncertain"}:
                untouched_reviewed += 1
                rows.append(row)
                continue

            source = (row.get("source") or "").strip().lower()
            rank = parse_rank(row.get("rank_info", ""))

            remove_it, reason = should_remove(source, rank)
            if remove_it:
                row["audit_decision"] = "remove"
                row["audit_reason"] = reason
                note = row.get("audit_notes", "") or ""
                row["audit_notes"] = (note + " | all-class-auto-marked").strip(" |")
                auto_removed += 1

            rows.append(row)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("All-class auto-mark complete.\n")
    print(f"Newly auto-marked remove rows: {auto_removed}")
    print(f"Already reviewed rows left untouched: {untouched_reviewed}")
    print(f"Updated file: {review_path}")


if __name__ == "__main__":
    main()
