import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor
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


def evaluate_single_model(
    model_name: ModelName,
    limit: int | None = None,
) -> dict:
    images_root = settings.FOOD101_IMAGES_DIR
    test_txt_path = settings.FOOD101_META_DIR / "test.txt"

    predictor = FoodPredictor()
    items = load_test_items(test_txt_path)
    if limit is not None:
        items = items[:limit]

    per_class = defaultdict(lambda: {"correct": 0, "total": 0, "mistakes": Counter()})

    for idx, (true_class, image_rel_path) in enumerate(items, start=1):
        image_path = images_root / image_rel_path
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        response = predictor.predict(image=image, model_name=model_name, top_k=1)
        predicted_class = response.predictions[0].class_name

        per_class[true_class]["total"] += 1
        if predicted_class == true_class:
            per_class[true_class]["correct"] += 1
        else:
            per_class[true_class]["mistakes"][predicted_class] += 1

        if idx % 250 == 0 or idx == len(items):
            print(f"[{model_name.value}] Evaluated {idx}/{len(items)} images...")

    report = []
    for class_name, stats in per_class.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else 0.0

        report.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 6),
                "top_mistakes": [
                    {"predicted_class": pred, "count": count}
                    for pred, count in stats["mistakes"].most_common(5)
                ],
            }
        )

    report.sort(key=lambda x: x["accuracy"])
    return {
        "model_name": model_name.value,
        "evaluated_items": sum(item["total"] for item in report),
        "per_class": report,
    }


def evaluate_ensemble(limit: int | None = None) -> dict:
    images_root = settings.FOOD101_IMAGES_DIR
    test_txt_path = settings.FOOD101_META_DIR / "test.txt"

    class_names = load_class_names(settings.CLASS_NAMES_PATH)
    loader = ModelLoader(num_classes=len(class_names))
    transforms = get_inference_transforms(image_size=224)

    eff_model = loader.load_model(ModelName.EFFICIENTNET_B0)
    res_model = loader.load_model(ModelName.RESNET50)

    items = load_test_items(test_txt_path)
    if limit is not None:
        items = items[:limit]

    per_class = defaultdict(lambda: {"correct": 0, "total": 0, "mistakes": Counter()})

    for idx, (true_class, image_rel_path) in enumerate(items, start=1):
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

        per_class[true_class]["total"] += 1
        if predicted_class == true_class:
            per_class[true_class]["correct"] += 1
        else:
            per_class[true_class]["mistakes"][predicted_class] += 1

        if idx % 250 == 0 or idx == len(items):
            print(f"[ensemble] Evaluated {idx}/{len(items)} images...")

    report = []
    for class_name, stats in per_class.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else 0.0

        report.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 6),
                "top_mistakes": [
                    {"predicted_class": pred, "count": count}
                    for pred, count in stats["mistakes"].most_common(5)
                ],
            }
        )

    report.sort(key=lambda x: x["accuracy"])
    return {
        "model_name": "ensemble_efficientnet_b0_resnet50",
        "evaluated_items": sum(item["total"] for item in report),
        "per_class": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-class accuracy.")
    parser.add_argument(
        "--mode",
        type=str,
        default="ensemble",
        choices=["ensemble", "efficientnet_b0", "resnet50", "mobilenet_v3_large"],
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "ensemble":
        report = evaluate_ensemble(limit=args.limit)
    else:
        report = evaluate_single_model(ModelName(args.mode), limit=args.limit)

    output_path = settings.MODEL_REPORTS_DIR / f"{args.mode}_per_class_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nPer-class evaluation complete.")
    print(f"Saved report to: {output_path}")

    print("\nWorst 15 classes:")
    for item in report["per_class"][:15]:
        print(
            f"{item['class_name']}: "
            f"accuracy={item['accuracy']:.4f} "
            f"({item['correct']}/{item['total']}) | "
            f"top mistakes={[m['predicted_class'] for m in item['top_mistakes']]}"
        )


if __name__ == "__main__":
    main()
