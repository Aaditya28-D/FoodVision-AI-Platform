import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


THUMB_SIZE = (180, 180)
PADDING = 16
TEXT_HEIGHT = 36
COLUMNS = 4
MAX_IMAGES_PER_CLASS = 24
BACKGROUND = (250, 250, 250)
TEXT_COLOR = (30, 30, 30)
BORDER_COLOR = (210, 210, 210)


def load_outlier_rows(csv_path: Path) -> dict[str, list[dict]]:
    rows_by_class: dict[str, list[dict]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_class[row["class_name"]].append(row)

    return rows_by_class


def fit_image_to_thumb(image: Image.Image, thumb_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(thumb_size)

    canvas = Image.new("RGB", thumb_size, (255, 255, 255))
    x = (thumb_size[0] - image.width) // 2
    y = (thumb_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def shorten_text(text: str, max_len: int = 24) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_contact_sheet(
    class_name: str,
    rows: list[dict],
    dataset_root: Path,
    output_path: Path,
) -> None:
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

    title = f"{class_name} — embedding outliers ({total} shown)"
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
        distance = row["distance_from_centroid"]
        rank = row["rank_within_class"]

        text_1 = shorten_text(f"#{rank} {filename}", 28)
        text_2 = shorten_text(f"dist={distance}", 28)

        draw.text((x0, y0 + THUMB_SIZE[1] + 6), text_1, fill=TEXT_COLOR, font=font)
        draw.text((x0, y0 + THUMB_SIZE[1] + 20), text_2, fill=(120, 80, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "data" / "food-101" / "images"
    outlier_csv = project_root / "backend" / "artifacts" / "dataset_cleaning" / "embedding_outliers.csv"
    output_dir = project_root / "backend" / "artifacts" / "dataset_cleaning" / "outlier_contact_sheets"

    if not outlier_csv.exists():
        raise FileNotFoundError(
            f"Missing embedding_outliers.csv. Run find_embedding_outliers.py first: {outlier_csv}"
        )

    rows_by_class = load_outlier_rows(outlier_csv)
    if not rows_by_class:
        raise RuntimeError("No outlier rows found in embedding_outliers.csv")

    for class_name, rows in sorted(rows_by_class.items()):
        output_path = output_dir / f"{class_name}_outliers.jpg"
        build_contact_sheet(
            class_name=class_name,
            rows=rows,
            dataset_root=dataset_root,
            output_path=output_path,
        )

    print("\nDone.")
    print(f"Outlier contact sheets saved under: {output_dir}")


if __name__ == "__main__":
    main()
