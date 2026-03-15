import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


def load_test_items(test_txt_path: Path) -> list[tuple[str, Path]]:
    items = []

    with test_txt_path.open("r", encoding="utf-8") as file:
        for line in file:
            rel_no_ext = line.strip()
            if not rel_no_ext:
                continue

            class_name = rel_no_ext.split("/")[0]
            image_rel_path = Path(f"{rel_no_ext}.jpg")
            items.append((class_name, image_rel_path))

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 + ResNet50 ensemble on Food-101 test set.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick evaluation.")
    args = parser.parse_args()

    images_root = settings.FOOD101_IMAGES_DIR
    test_txt_path = settings.FOOD101_META_DIR / "test.txt"
    output_path = settings.MODEL_REPORTS_DIR / "ensemble_evaluation_report.json"

    class_names = load_class_names(settings.CLASS_NAMES_PATH)
    loader = ModelLoader(num_classes=len(class_names))
    transforms = get_inference_transforms(image_size=224)

    eff_model = loader.load_model(ModelName.EFFICIENTNET_B0)
    res_model = loader.load_model(ModelName.RESNET50)

    test_items = load_test_items(test_txt_path)
    if args.limit is not None:
        test_items = test_items[: args.limit]

    correct = 0
    total = 0
    mistakes = Counter()

    for idx, (true_class, image_rel_path) in enumerate(test_items, start=1):
        image_path = images_root / image_rel_path
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms(image).unsqueeze(0).to(eff_model.device)

        with torch.no_grad():
            eff_logits = eff_model.model(image_tensor)
            res_logits = res_model.model(image_tensor)

            eff_probs = torch.softmax(eff_logits, dim=1)
            res_probs = torch.softmax(res_logits, dim=1)

            ensemble_probs = (eff_probs + res_probs) / 2.0
            pred_idx = torch.argmax(ensemble_probs, dim=1).item()

        predicted_class = class_names[pred_idx]

        total += 1
        if predicted_class == true_class:
            correct += 1
        else:
            mistakes[(true_class, predicted_class)] += 1

        if idx % 250 == 0 or idx == len(test_items):
            print(f"Evaluated {idx}/{len(test_items)} images...")

    accuracy = correct / total if total else 0.0

    report = {
        "ensemble": "efficientnet_b0 + resnet50",
        "evaluated_items": total,
        "correct": correct,
        "top1_accuracy": round(accuracy, 6),
        "top_mistakes": [
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": count,
            }
            for (true_class, predicted_class), count in mistakes.most_common(25)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nEnsemble evaluation complete.")
    print(f"Saved report to: {output_path}")
    print(f"Ensemble accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
