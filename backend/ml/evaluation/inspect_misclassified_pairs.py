from pathlib import Path
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt

from ml.inference.predictor import FoodPredictor
from ml.inference.model_registry import ModelName


def load_items(split_path: Path) -> list[tuple[str, Path]]:
    items = []
    with split_path.open("r", encoding="utf-8") as file:
        for line in file:
            rel_no_ext = line.strip()
            if not rel_no_ext:
                continue
            class_name = rel_no_ext.split("/")[0]
            image_rel_path = Path(f"{rel_no_ext}.jpg")
            items.append((class_name, image_rel_path))
    return items


def save_grid(image_entries: list[tuple[Path, str]], title: str, output_path: Path) -> None:
    cols = 4
    rows = (len(image_entries) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, (image_path, subtitle) in zip(axes, image_entries):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(subtitle, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    images_root = project_root / "data" / "food-101" / "images"
    test_txt_path = project_root / "data" / "food-101" / "meta" / "test.txt"
    output_dir = project_root / "backend" / "artifacts" / "confusion_inspection"

    predictor = FoodPredictor()
    items = load_items(test_txt_path)

    target_pairs = {
        ("cheese_plate", "cheesecake"): [],
        ("cheesecake", "cheese_plate"): [],
    }

    counts = defaultdict(int)

    for idx, (true_class, image_rel_path) in enumerate(items, start=1):
        image_path = images_root / image_rel_path
        if not image_path.exists():
            continue

        if true_class not in {"cheese_plate", "cheesecake"}:
            continue

        image = Image.open(image_path).convert("RGB")
        response = predictor.predict(
            image=image,
            model_name=ModelName.EFFICIENTNET_B0,
            top_k=1,
        )
        predicted_class = response.predictions[0].class_name

        pair = (true_class, predicted_class)
        counts[pair] += 1

        if pair in target_pairs and len(target_pairs[pair]) < 12:
            subtitle = f"{image_path.name}\npred={predicted_class}"
            target_pairs[pair].append((image_path, subtitle))

        if idx % 100 == 0:
            print(f"Checked {idx} relevant images...")

    for (true_class, predicted_class), entries in target_pairs.items():
        if entries:
            save_grid(
                entries,
                title=f"Misclassified: {true_class} -> {predicted_class}",
                output_path=output_dir / f"{true_class}_to_{predicted_class}.png",
            )

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")
    print("\nCounts seen in test set for these directions:")
    for pair, count in sorted(counts.items()):
        if pair[0] in {"cheese_plate", "cheesecake"}:
            print(f"{pair[0]} -> {pair[1]}: {count}")


if __name__ == "__main__":
    main()