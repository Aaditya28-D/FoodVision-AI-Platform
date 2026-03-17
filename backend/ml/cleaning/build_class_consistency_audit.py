import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_CLASSES = [
    "pizza",
    "cheese_plate",
    "cheesecake",
    "steak",
    "filet_mignon",
    "prime_rib",
    "pork_chop",
    "hamburger",
    "hot_dog",
    "tacos",
    "pho",
]

THUMB_SIZE = (180, 180)
PADDING = 16
TEXT_HEIGHT = 52
COLUMNS = 4
BACKGROUND = (250, 250, 250)
TEXT_COLOR = (30, 30, 30)
BORDER_COLOR = (210, 210, 210)

NUM_REMOVED = 8
NUM_OUTLIERS = 8
NUM_RANDOM = 8
RANDOM_SEED = 42


def shorten_text(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def fit_image_to_thumb(image: Image.Image, thumb_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(thumb_size)

    canvas = Image.new("RGB", thumb_size, (255, 255, 255))
    x = (thumb_size[0] - image.width) // 2
    y = (thumb_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def read_remove_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def read_outlier_csv(path: Path) -> dict[str, list[dict]]:
    rows_by_class = defaultdict(list)
    if not path.exists():
        return rows_by_class

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_class[row["class_name"]].append(row)

    for class_name in rows_by_class:
        rows_by_class[class_name].sort(key=lambda x: int(x["rank_within_class"]))

    return rows_by_class


def collect_random_samples(dataset_root: Path, class_name: str, n: int, exclude: set[str]) -> list[str]:
    class_dir = dataset_root / class_name
    if not class_dir.exists():
        return []

    all_paths = sorted(str(p.relative_to(dataset_root)) for p in class_dir.glob("*.jpg"))
    available = [p for p in all_paths if p not in exclude]

    if not available:
        return []

    random.shuffle(available)
    return available[:n]


def build_rows_for_class(
    class_name: str,
    dataset_root: Path,
    remove_set: set[str],
    outliers_by_class: dict[str, list[dict]],
) -> list[dict]:
    rows = []
    used = set()

    removed_for_class = sorted([p for p in remove_set if p.startswith(f"{class_name}/")])[:NUM_REMOVED]
    for image_path in removed_for_class:
        rows.append(
            {
                "class_name": class_name,
                "image_path": image_path,
                "source": "removed",
                "rank_info": "",
            }
        )
        used.add(image_path)

    outlier_rows = outliers_by_class.get(class_name, [])
    for row in outlier_rows:
        image_path = row["image_path"]
        if image_path in used:
            continue
        rows.append(
            {
                "class_name": class_name,
                "image_path": image_path,
                "source": "outlier",
                "rank_info": f"rank={row['rank_within_class']}",
            }
        )
        used.add(image_path)
        if sum(1 for r in rows if r["source"] == "outlier") >= NUM_OUTLIERS:
            break

    random_rows = collect_random_samples(dataset_root, class_name, NUM_RANDOM, used)
    for image_path in random_rows:
        rows.append(
            {
                "class_name": class_name,
                "image_path": image_path,
                "source": "random",
                "rank_info": "",
            }
        )

    return rows


def build_contact_sheet(class_name: str, rows: list[dict], dataset_root: Path, output_path: Path) -> None:
    if not rows:
        return

    font = ImageFont.load_default()
    total = len(rows)
    cols = COLUMNS
    grid_rows = math.ceil(total / cols)

    cell_width = THUMB_SIZE[0] + PADDING * 2
    cell_height = THUMB_SIZE[1] + TEXT_HEIGHT + PADDING * 2

    title_height = 60
    width = cols * cell_width
    height = title_height + grid_rows * cell_height

    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    title = f"{class_name} — class consistency audit"
    draw.text((PADDING, 18), title, fill=TEXT_COLOR, font=font)

    for idx, row in enumerate(rows):
        rel_path = Path(row["image_path"])
        image_path = dataset_root / rel_path

        col = idx % cols
        row_idx = idx // cols

        x0 = col * cell_width + PADDING
        y0 = title_height + row_idx * cell_height + PADDING
        thumb_box = (x0, y0, x0 + THUMB_SIZE[0], y0 + THUMB_SIZE[1])

        try:
            with Image.open(image_path) as img:
                thumb = fit_image_to_thumb(img, THUMB_SIZE)
        except Exception:
            thumb = Image.new("RGB", THUMB_SIZE, (245, 220, 220))
            err_draw = ImageDraw.Draw(thumb)
            err_draw.text((10, 80), "read error", fill=(120, 0, 0), font=font)

        sheet.paste(thumb, (x0, y0))
        draw.rectangle(thumb_box, outline=BORDER_COLOR, width=1)

        source = row["source"]
        filename = rel_path.name
        rank_info = row["rank_info"]

        draw.text((x0, y0 + THUMB_SIZE[1] + 6), shorten_text(filename), fill=TEXT_COLOR, font=font)
        draw.text((x0, y0 + THUMB_SIZE[1] + 22), shorten_text(f"source={source}"), fill=(0, 90, 140), font=font)
        draw.text((x0, y0 + THUMB_SIZE[1] + 38), shorten_text(rank_info), fill=(120, 80, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def save_audit_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "image_path",
                "source",
                "rank_info",
                "audit_decision",
                "audit_reason",
                "audit_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "data" / "food-101" / "images"
    cleaning_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"

    remove_manifest = cleaning_dir / "cleaned_remove_manifest.txt"
    outlier_csv = cleaning_dir / "embedding_outliers.csv"

    output_dir = cleaning_dir / "class_consistency_audit"
    audit_csv_path = output_dir / "class_consistency_audit_candidates.csv"

    remove_set = read_remove_manifest(remove_manifest)
    outliers_by_class = read_outlier_csv(outlier_csv)

    all_rows = []

    for class_name in TARGET_CLASSES:
        class_rows = build_rows_for_class(class_name, dataset_root, remove_set, outliers_by_class)
        for row in class_rows:
            row["audit_decision"] = ""
            row["audit_reason"] = ""
            row["audit_notes"] = ""
        all_rows.extend(class_rows)

        sheet_path = output_dir / f"{class_name}_class_consistency_audit.jpg"
        build_contact_sheet(class_name, class_rows, dataset_root, sheet_path)

    save_audit_csv(all_rows, audit_csv_path)

    print("\nDone.")
    print(f"Saved audit CSV: {audit_csv_path}")
    print(f"Saved audit sheets under: {output_dir}")


if __name__ == "__main__":
    main()
