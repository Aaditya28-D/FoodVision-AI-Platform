import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def variance_of_laplacian(gray_array: np.ndarray) -> float:
    gray = gray_array.astype(np.float32)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(lap.var())


def analyze_image(image_path: Path) -> dict:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        aspect_ratio = width / height if height else 0.0

        gray = img.convert("L")
        gray_np = np.array(gray)

        sharpness_score = variance_of_laplacian(gray_np)

        rgb_stat = ImageStat.Stat(img)
        grayscale_std = float(np.std(gray_np))
        brightness_mean = float(np.mean(gray_np))

        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 4),
            "sharpness_score": round(sharpness_score, 4),
            "grayscale_std": round(grayscale_std, 4),
            "brightness_mean": round(brightness_mean, 4),
        }


def build_flags(stats: dict) -> list[str]:
    flags = []

    width = stats["width"]
    height = stats["height"]
    aspect_ratio = stats["aspect_ratio"]
    sharpness_score = stats["sharpness_score"]
    grayscale_std = stats["grayscale_std"]
    brightness_mean = stats["brightness_mean"]

    if width < 256 or height < 256:
        flags.append("small_image")

    if aspect_ratio > 1.8 or aspect_ratio < 0.55:
        flags.append("extreme_aspect_ratio")

    if sharpness_score < 20:
        flags.append("very_blurry")
    elif sharpness_score < 40:
        flags.append("blurry")

    if grayscale_std < 18:
        flags.append("low_contrast")

    if brightness_mean < 35:
        flags.append("too_dark")
    elif brightness_mean > 220:
        flags.append("too_bright")

    return flags


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "data" / "food-101" / "images"
    output_dir = project_root / "backend" / "artifacts" / "dataset_cleaning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "suspicious_images.csv"
    summary_csv = output_dir / "suspicious_summary_by_class.csv"

    image_paths = sorted(dataset_root.glob("*/*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {dataset_root}")

    rows = []
    class_flag_counts = defaultdict(int)
    class_total_suspicious = defaultdict(int)

    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        class_name = image_path.parent.name
        rel_path = image_path.relative_to(dataset_root)

        try:
            stats = analyze_image(image_path)
            flags = build_flags(stats)
        except Exception as exc:
            flags = ["image_read_error"]
            stats = {
                "width": "",
                "height": "",
                "aspect_ratio": "",
                "sharpness_score": "",
                "grayscale_std": "",
                "brightness_mean": "",
            }

        if flags:
            rows.append(
                {
                    "class_name": class_name,
                    "image_path": str(rel_path),
                    "flag_count": len(flags),
                    "flags": "|".join(flags),
                    **stats,
                }
            )

            class_total_suspicious[class_name] += 1
            for flag in flags:
                class_flag_counts[(class_name, flag)] += 1

        if idx % 5000 == 0 or idx == total:
            print(f"Processed {idx}/{total} images...")

    rows.sort(key=lambda x: (x["class_name"], -x["flag_count"], x["image_path"]))

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    classes = sorted({row["class_name"] for row in rows})

    for class_name in classes:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        summary_rows.append(
            {
                "class_name": class_name,
                "suspicious_images": len(class_rows),
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "suspicious_images"],
        )
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda x: x["suspicious_images"], reverse=True))

    print("\nDone.")
    print(f"Saved suspicious image report to: {output_csv}")
    print(f"Saved summary by class to: {summary_csv}")
    print(f"Total suspicious images flagged: {len(rows)}")


if __name__ == "__main__":
    main()
