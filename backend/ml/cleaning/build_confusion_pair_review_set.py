import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_CLASSES = {
    "cheese_plate",
    "cheesecake",
    "steak",
    "filet_mignon",
    "prime_rib",
    "pork_chop",
    "beef_tartare",
    "tuna_tartare",
    "chocolate_mousse",
    "chocolate_cake",
    "ravioli",
    "gnocchi",
    "apple_pie",
    "bread_pudding",
}

THUMB_SIZE = (180, 180)
PADDING = 16
TEXT_HEIGHT = 42
COLUMNS = 4
MAX_IMAGES_PER_CLASS = 24
BACKGROUND = (250, 250, 250)
TEXT_COLOR = (30, 30, 30)
BORDER_COLOR = (210, 210, 210)


def shorten_text(text: str, max_len: int = 26) -> str:
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


def load_embedding_index(index_path: Path):
    data = np.load(index_path, allow_pickle=True)
    return data["embeddings"], data["image_paths"], data["class_names"]


def build_outlier_rows(embeddings, image_paths, class_names):
    class_to_indices = defaultdict(list)
    for idx, class_name in enumerate(class_names):
        class_name = str(class_name)
        if class_name in TARGET_CLASSES:
            class_to_indices[class_name].append(idx)

    rows = []
    for class_name, indices in sorted(class_to_indices.items()):
        class_embeddings = embeddings[indices]
        if len(class_embeddings) < 2:
            continue

        centroid = class_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        distances = []
        for local_idx, global_idx in enumerate(indices):
            emb = class_embeddings[local_idx]
            dist = float(np.linalg.norm(emb - centroid))
            distances.append((global_idx, dist))

        distances.sort(key=lambda x: x[1], reverse=True)

        for rank, (global_idx, distance) in enumerate(distances[:50], start=1):
            rows.append(
                {
                    "class_name": class_name,
                    "image_path": str(image_paths[global_idx]),
                    "rank_within_class": rank,
                    "distance_from_centroid": round(distance, 6),
                    "review_decision": "",
                    "review_reason": "",
                    "review_notes": "",
                }
            )

    return rows


def save_review_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "image_path",
                "rank_within_class",
                "distance_from_centroid",
                "review_decision",
                "review_reason",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(class_name: str, rows: list[dict], dataset_root: Path, output_path: Path) -> None:
    rows = rows[:MAX_IMAGES_PER_CLASS]
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

    title = f"{class_name} — confusion review candidates ({total} shown)"
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

        filename = rel_path.name
        rank = row["rank_within_class"]
        dist = row["distance_from_centroid"]

        draw.text((x0, y0 + THUMB_SIZE[1] + 6), shorten_text(f"#{rank} {filename}", 26), fill=TEXT_COLOR, font=font)
        draw.text((x0, y0 + THUMB_SIZE[1] + 22), shorten_text(f"dist={dist}", 26), fill=(120, 80, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "data" / "food-101" / "images"
    index_path = project_root / "data" / "embeddings" / "food101_resnet50_index.npz"
    output_dir = project_root / "backend" / "artifacts" / "dataset_cleaning" / "confusion_pair_review"

    embeddings, image_paths, class_names = load_embedding_index(index_path)
    rows = build_outlier_rows(embeddings, image_paths, class_names)

    review_csv_path = output_dir / "confusion_pair_review_candidates.csv"
    save_review_csv(rows, review_csv_path)

    rows_by_class = defaultdict(list)
    for row in rows:
        rows_by_class[row["class_name"]].append(row)

    for class_name, class_rows in sorted(rows_by_class.items()):
        sheet_path = output_dir / f"{class_name}_confusion_review.jpg"
        build_contact_sheet(class_name, class_rows, dataset_root, sheet_path)

    print("\nDone.")
    print(f"Saved review CSV: {review_csv_path}")
    print(f"Saved contact sheets under: {output_dir}")


if __name__ == "__main__":
    main()
