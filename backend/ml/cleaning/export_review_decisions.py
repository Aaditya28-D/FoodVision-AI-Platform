import csv
from collections import Counter, defaultdict
from pathlib import Path


VALID_DECISIONS = {"keep", "remove", "uncertain", ""}


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    review_csv = cleaning_dir / "review_candidates.csv"
    keep_txt = cleaning_dir / "keep_list.txt"
    remove_txt = cleaning_dir / "remove_list.txt"
    uncertain_txt = cleaning_dir / "uncertain_list.txt"
    summary_class_csv = cleaning_dir / "review_summary_by_class.csv"
    summary_reason_csv = cleaning_dir / "review_summary_by_reason.csv"

    if not review_csv.exists():
        raise FileNotFoundError(
            f"Missing review_candidates.csv: {review_csv}"
        )

    keep_paths: list[str] = []
    remove_paths: list[str] = []
    uncertain_paths: list[str] = []

    class_summary = defaultdict(lambda: {"keep": 0, "remove": 0, "uncertain": 0, "unreviewed": 0})
    reason_counter = Counter()

    total_rows = 0

    with review_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            class_name = row["class_name"]
            image_path = row["image_path"]
            decision = (row.get("review_decision") or "").strip().lower()
            reason = (row.get("review_reason") or "").strip().lower()

            if decision not in VALID_DECISIONS:
                raise ValueError(
                    f"Invalid review_decision '{decision}' for image '{image_path}'. "
                    f"Allowed: keep, remove, uncertain, blank"
                )

            if decision == "keep":
                keep_paths.append(image_path)
                class_summary[class_name]["keep"] += 1
            elif decision == "remove":
                remove_paths.append(image_path)
                class_summary[class_name]["remove"] += 1
            elif decision == "uncertain":
                uncertain_paths.append(image_path)
                class_summary[class_name]["uncertain"] += 1
            else:
                class_summary[class_name]["unreviewed"] += 1

            if reason:
                reason_counter[reason] += 1

    keep_paths.sort()
    remove_paths.sort()
    uncertain_paths.sort()

    keep_txt.write_text("\n".join(keep_paths) + ("\n" if keep_paths else ""), encoding="utf-8")
    remove_txt.write_text("\n".join(remove_paths) + ("\n" if remove_paths else ""), encoding="utf-8")
    uncertain_txt.write_text("\n".join(uncertain_paths) + ("\n" if uncertain_paths else ""), encoding="utf-8")

    with summary_class_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "keep", "remove", "uncertain", "unreviewed", "reviewed_total"],
        )
        writer.writeheader()

        for class_name in sorted(class_summary.keys()):
            item = class_summary[class_name]
            reviewed_total = item["keep"] + item["remove"] + item["uncertain"]
            writer.writerow(
                {
                    "class_name": class_name,
                    "keep": item["keep"],
                    "remove": item["remove"],
                    "uncertain": item["uncertain"],
                    "unreviewed": item["unreviewed"],
                    "reviewed_total": reviewed_total,
                }
            )

    with summary_reason_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["review_reason", "count"],
        )
        writer.writeheader()

        for reason, count in reason_counter.most_common():
            writer.writerow(
                {
                    "review_reason": reason,
                    "count": count,
                }
            )

    print("Export complete.\n")
    print(f"Total rows in review CSV: {total_rows}")
    print(f"Keep: {len(keep_paths)}")
    print(f"Remove: {len(remove_paths)}")
    print(f"Uncertain: {len(uncertain_paths)}")
    print(f"Unreviewed: {total_rows - len(keep_paths) - len(remove_paths) - len(uncertain_paths)}\n")

    print(f"Saved: {keep_txt}")
    print(f"Saved: {remove_txt}")
    print(f"Saved: {uncertain_txt}")
    print(f"Saved: {summary_class_csv}")
    print(f"Saved: {summary_reason_csv}")


if __name__ == "__main__":
    main()
