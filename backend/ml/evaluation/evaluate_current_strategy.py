import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor


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
    parser = argparse.ArgumentParser(
        description="Evaluate current FoodVision prediction strategy on Food-101 test set."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smart",
        choices=["smart", "ensemble", "mobilenet", "efficientnet", "resnet"],
        help="Prediction strategy to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick evaluation.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data" / "food-101"
    images_root = data_root / "images"
    test_txt_path = data_root / "meta" / "test.txt"
    output_path = project_root / "backend" / "models" / f"evaluation_{args.mode}.json"

    test_items = load_test_items(test_txt_path)
    if args.limit is not None:
        test_items = test_items[: args.limit]

    if not test_items:
        raise RuntimeError("No test items found to evaluate.")

    predictor = FoodPredictor()

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

    for idx, (true_class, image_rel_path) in enumerate(test_items, start=1):
        image_path = images_root / image_rel_path
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        if args.mode == "smart":
            response = predictor.predict_smart(image=image, top_k=1)
        elif args.mode == "ensemble":
            response = predictor.predict_ensemble(image=image, top_k=1)
        elif args.mode == "mobilenet":
            response = predictor.predict(
                image=image,
                model_name=ModelName.MOBILENET_V3_LARGE,
                top_k=1,
            )
        elif args.mode == "efficientnet":
            response = predictor.predict(
                image=image,
                model_name=ModelName.EFFICIENTNET_B0,
                top_k=1,
            )
        elif args.mode == "resnet":
            response = predictor.predict(
                image=image,
                model_name=ModelName.RESNET50,
                top_k=1,
            )
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        predicted_class = response.predictions[0].class_name

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
        "model_name": args.mode,
        "evaluated_items": stats["total"],
        "correct": stats["correct"],
        "top1_accuracy": round(accuracy, 6),
        "top_mistakes": top_mistakes,
        "per_class": per_class_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nEvaluation complete.")
    print(f"Saved report to: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Top-1 accuracy: {accuracy:.6f} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
