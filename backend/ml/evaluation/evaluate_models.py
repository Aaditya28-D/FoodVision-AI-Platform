import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

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
    parser = argparse.ArgumentParser(description="Evaluate current FoodVision models on Food-101 test set.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick evaluation.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data" / "food-101"
    images_root = data_root / "images"
    test_txt_path = data_root / "meta" / "test.txt"
    output_path = project_root / "backend" / "models" / "evaluation_report.json"

    test_items = load_test_items(test_txt_path)
    if args.limit is not None:
        test_items = test_items[: args.limit]

    if not test_items:
        raise RuntimeError("No test items found to evaluate.")

    predictor = FoodPredictor()

    model_stats: dict[str, dict] = defaultdict(
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
        comparison = predictor.compare_models(image=image, top_k=1)

        for result in comparison.results:
            model_name = result.model_name
            predicted_class = result.top_prediction.class_name

            model_stats[model_name]["total"] += 1

            if predicted_class == true_class:
                model_stats[model_name]["correct"] += 1
            else:
                model_stats[model_name]["mistakes"][(true_class, predicted_class)] += 1

        if idx % 250 == 0 or idx == total_items:
            print(f"Evaluated {idx}/{total_items} images...")

    report = {
        "evaluated_items": total_items,
        "models": {},
    }

    for model_name, stats in model_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else 0.0

        top_mistakes = [
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": count,
            }
            for (true_class, predicted_class), count in stats["mistakes"].most_common(25)
        ]

        report["models"][model_name] = {
            "correct": correct,
            "total": total,
            "top1_accuracy": round(accuracy, 6),
            "top_mistakes": top_mistakes,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nEvaluation complete.")
    print(f"Saved report to: {output_path}\n")

    for model_name, stats in report["models"].items():
        print(
            f"{model_name}: "
            f"accuracy={stats['top1_accuracy']:.4f} "
            f"({stats['correct']}/{stats['total']})"
        )


if __name__ == "__main__":
    main()