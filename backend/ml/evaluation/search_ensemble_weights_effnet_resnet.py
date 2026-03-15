import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Search weighted ensemble for EfficientNet + ResNet.")
    parser.add_argument("--limit", type=int, default=2000, help="Optional limit for faster search.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data" / "food-101"
    images_root = data_root / "images"
    test_txt_path = data_root / "meta" / "test.txt"
    output_path = project_root / "backend" / "models" / "reports" / "weight_search_effnet_resnet.json"

    class_names = load_class_names(settings.CLASS_NAMES_PATH)
    loader = ModelLoader(num_classes=len(class_names))
    transforms = get_inference_transforms(image_size=224)

    eff_model = loader.load_model(ModelName.EFFICIENTNET_B0)
    res_model = loader.load_model(ModelName.RESNET50)

    items = load_test_items(test_txt_path)
    if args.limit is not None:
        items = items[: args.limit]

    weights = [round(i / 10, 1) for i in range(1, 10)]
    results = []

    for eff_weight in weights:
        res_weight = round(1.0 - eff_weight, 1)
        correct = 0
        total = 0

        print(f"\nTesting weights: eff={eff_weight}, res={res_weight}")

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

                ensemble_probs = (eff_weight * eff_probs) + (res_weight * res_probs)
                pred_idx = torch.argmax(ensemble_probs, dim=1).item()

            predicted_class = class_names[pred_idx]

            total += 1
            if predicted_class == true_class:
                correct += 1

            if idx % 250 == 0 or idx == len(items):
                print(f"  Evaluated {idx}/{len(items)} images...")

        accuracy = correct / total if total else 0.0
        results.append(
            {
                "efficientnet_weight": eff_weight,
                "resnet_weight": res_weight,
                "correct": correct,
                "total": total,
                "top1_accuracy": round(accuracy, 6),
            }
        )

        print(f"  Accuracy: {accuracy:.6f} ({correct}/{total})")

    results.sort(key=lambda x: x["top1_accuracy"], reverse=True)

    report = {
        "search_name": "effnet_resnet_weight_search",
        "evaluated_items": results[0]["total"] if results else 0,
        "results": results,
        "best": results[0] if results else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nWeight search complete.")
    print(f"Saved report to: {output_path}")
    if results:
        best = results[0]
        print(
            f"Best weights -> eff={best['efficientnet_weight']}, "
            f"res={best['resnet_weight']}, "
            f"accuracy={best['top1_accuracy']} ({best['correct']}/{best['total']})"
        )


if __name__ == "__main__":
    main()
