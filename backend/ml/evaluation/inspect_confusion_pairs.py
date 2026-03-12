from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt


def load_sample_paths(class_dir: Path, n: int = 12) -> list[Path]:
    image_paths = sorted(class_dir.glob("*.jpg"))
    if not image_paths:
        return []
    random.seed(42)
    if len(image_paths) <= n:
        return image_paths
    return random.sample(image_paths, n)


def make_grid(image_paths: list[Path], title: str, output_path: Path) -> None:
    cols = 4
    rows = (len(image_paths) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, image_path in zip(axes, image_paths):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    images_root = project_root / "data" / "food-101" / "images"
    out_dir = project_root / "backend" / "artifacts" / "confusion_inspection"

    class_a = "cheese_plate"
    class_b = "cheesecake"

    class_a_dir = images_root / class_a
    class_b_dir = images_root / class_b

    if not class_a_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_a_dir}")
    if not class_b_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_b_dir}")

    a_paths = load_sample_paths(class_a_dir, n=12)
    b_paths = load_sample_paths(class_b_dir, n=12)

    make_grid(
        a_paths,
        title=f"Sample images: {class_a}",
        output_path=out_dir / f"{class_a}_samples.png",
    )

    make_grid(
        b_paths,
        title=f"Sample images: {class_b}",
        output_path=out_dir / f"{class_b}_samples.png",
    )

    print(f"Saved: {out_dir / f'{class_a}_samples.png'}")
    print(f"Saved: {out_dir / f'{class_b}_samples.png'}")


if __name__ == "__main__":
    main()