import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.transforms import get_inference_transforms
from ml.models.efficientnet import build_efficientnet_b0


def load_test_items(test_txt_path: Path) -> list[tuple[str, Path]]:
    if not test_txt_path.exists():
        raise FileNotFoundError(f"Test split file not found: {test_txt_path}")

    items: list[tuple[str, Path]] = []

    with test_txt_path.open("r", encoding="utf-8") as file:
        for line in file:
            rel_no_ext = line.strip()
            if not rel_no_ext:
                continue

            class_name = rel_no_ext.split("/")[0]
            image_rel_path = Path(f"{rel_no_ext}.jpg")
            items.append((class_name, image_rel_path))

    return items


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data" / "food-101"
    images_root = data_root / "images"
    test_txt_path = data_root / "meta" / "test.txt"

    weights_path = project_root / "backend" / "models" / "efficientnet_b0_cleaned_best.pth"
    report_path = project_root / "backend" / "models" / "reports" / "evaluation_efficientnet_b0_cleaned.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Cleaned model weights not found: {weights_path}")

    class_names = load_class_names(settings.CLASS_NAMES_PATH)
    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    device = resolve_device()
    model = build_efficientnet_b0(num_classes=num_classes)
    loaded = torch.load(weights_path, map_location=device)

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        state_dict = loaded

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transforms = get_inference_transforms(image_size=224)
    test_items = load_test_items(test_txt_path)

    stats = {
        "correct": 0,
        "total": 0,
        "mistakes": Counter(),
    }

    per_class = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
            "mistakes": Counter(),
        }
    )

    total_items = len(test_items)

    with torch.no_grad():
        for idx, (true_class, image_rel_path) in enumerate(test_items, start=1):
            image_path = images_root / image_rel_path
            if not image_path.exists():
                print(f"Skipping missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            image_tensor = transforms(image).unsqueeze(0).to(device)

            logits = model(image_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            predicted_class = class_names[pred_idx]

            stats["total"] += 1
            per_class[true_class]["total"] += 1

            if predicted_class == true_class:
                stats["correct"] += 1
                per_class[true_class]["correct"] += 1
            else:
                stats["mistakes"][(true_class, predicted_class)] += 1
                per_class[true_class]["mistakes"][predicted_class] += 1

            if idx % 250 == 0 or idx == total_items:
                print(f"Evaluated {idx}/{total_items} images...")

    accuracy = stats["correct"] / stats["total"] if stats["total"] else 0.0

    top_mistakes = [
        {
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": count,
        }
        for (true_class, predicted_class), count in stats["mistakes"].most_common(25)
    ]

    per_class_rows = []
    for class_name, class_stats in per_class.items():
        class_total = class_stats["total"]
        class_correct = class_stats["correct"]
        class_accuracy = class_correct / class_total if class_total else 0.0

        class_top_mistakes = [
            {
                "predicted_class": predicted_class,
                "count": count,
            }
            for predicted_class, count in class_stats["mistakes"].most_common(5)
        ]

        per_class_rows.append(
            {
                "class_name": class_name,
                "correct": class_correct,
                "total": class_total,
                "accuracy": round(class_accuracy, 6),
                "top_mistakes": class_top_mistakes,
            }
        )

    per_class_rows.sort(key=lambda x: x["accuracy"])

    report = {
        "model_name": "efficientnet_b0_cleaned",
        "weights_path": str(weights_path),
        "evaluated_items": stats["total"],
        "correct": stats["correct"],
        "top1_accuracy": round(accuracy, 6),
        "top_mistakes": top_mistakes,
        "per_class": per_class_rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nEvaluation complete.")
    print(f"Saved report to: {report_path}")
    print(f"Model: efficientnet_b0_cleaned")
    print(f"Top-1 accuracy: {accuracy:.6f} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
