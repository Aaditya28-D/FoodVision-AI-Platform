import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ml.inference.predictor import FoodPredictor
from ml.inference.model_registry import ModelName


DATASET_ROOT = Path("../data/food-101")
TEST_ROOT = DATASET_ROOT / "images"
TEST_SPLIT_FILE = DATASET_ROOT / "meta" / "test.txt"
OUTPUT_PATH = Path("../models/evaluation_smart_v2.json")


FALLBACK_CLASS_MODEL_MAP = {
    "cheesecake": ModelName.MOBILENET_V3_LARGE,
    "cheese_plate": ModelName.MOBILENET_V3_LARGE,
    "pork_chop": ModelName.EFFICIENTNET_B0,
    "bread_pudding": ModelName.EFFICIENTNET_B0,
    "apple_pie": ModelName.EFFICIENTNET_B0,
    "foie_gras": ModelName.RESNET50,
    "ravioli": ModelName.RESNET50,
    "chocolate_cake": ModelName.RESNET50,
    "tuna_tartare": ModelName.RESNET50,
}

STRONG_CONFIDENCE_GAP = 0.15
LOW_CONFIDENCE_THRESHOLD = 0.55


def choose_prediction(results: dict):
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["confidence"],
        reverse=True,
    )

    best_model_name, best_pred = sorted_models[0]
    second_model_name, second_pred = sorted_models[1]

    top_classes = [item["class_name"] for item in results.values()]
    class_counts = Counter(top_classes)

    if (
        best_pred["confidence"] >= LOW_CONFIDENCE_THRESHOLD
        and (best_pred["confidence"] - second_pred["confidence"]) >= STRONG_CONFIDENCE_GAP
    ):
        return best_pred["class_name"], best_model_name, "strong_confidence"

    majority_label, majority_count = class_counts.most_common(1)[0]
    if majority_count >= 2:
        return majority_label, "majority_vote", "majority_vote"

    fallback_class = best_pred["class_name"]
    preferred_model = FALLBACK_CLASS_MODEL_MAP.get(fallback_class, ModelName.EFFICIENTNET_B0)

    preferred_model_name = preferred_model.value
    preferred_prediction = results[preferred_model_name]

    return preferred_prediction["class_name"], preferred_model_name, "fallback_map"


def main():
    predictor = FoodPredictor()

    with open(TEST_SPLIT_FILE, "r", encoding="utf-8") as f:
        test_items = [line.strip() for line in f if line.strip()]

    image_paths = [TEST_ROOT / f"{item}.jpg" for item in test_items]

    total = 0
    correct = 0

    mistake_counter = Counter()
    per_class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "mistakes": Counter()})

    for image_path in tqdm(image_paths, desc="Evaluating smart_v2"):
        true_class = image_path.parent.name

        image = Image.open(image_path).convert("RGB")

        eff = predictor.predict(image=image, model_name=ModelName.EFFICIENTNET_B0, top_k=1)
        res = predictor.predict(image=image, model_name=ModelName.RESNET50, top_k=1)
        mob = predictor.predict(image=image, model_name=ModelName.MOBILENET_V3_LARGE, top_k=1)

        results = {
            "efficientnet_b0": {
                "class_name": eff.predictions[0].class_name,
                "confidence": eff.predictions[0].confidence,
            },
            "resnet50": {
                "class_name": res.predictions[0].class_name,
                "confidence": res.predictions[0].confidence,
            },
            "mobilenet_v3_large": {
                "class_name": mob.predictions[0].class_name,
                "confidence": mob.predictions[0].confidence,
            },
        }

        predicted_class, chosen_source, decision_type = choose_prediction(results)

        total += 1
        per_class_stats[true_class]["total"] += 1

        if predicted_class == true_class:
            correct += 1
            per_class_stats[true_class]["correct"] += 1
        else:
            mistake_counter[(true_class, predicted_class)] += 1
            per_class_stats[true_class]["mistakes"][predicted_class] += 1

    top1_accuracy = correct / total if total else 0.0

    top_mistakes = [
        {
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": count,
        }
        for (true_class, predicted_class), count in mistake_counter.most_common(25)
    ]

    per_class = []
    for class_name, stats in per_class_stats.items():
        class_total = stats["total"]
        class_correct = stats["correct"]
        class_accuracy = class_correct / class_total if class_total else 0.0

        per_class.append(
            {
                "class_name": class_name,
                "correct": class_correct,
                "total": class_total,
                "accuracy": round(class_accuracy, 6),
                "top_mistakes": [
                    {
                        "predicted_class": pred_class,
                        "count": count,
                    }
                    for pred_class, count in stats["mistakes"].most_common(5)
                ],
            }
        )

    per_class.sort(key=lambda x: x["accuracy"])

    output = {
        "model_name": "smart_v2",
        "evaluated_items": total,
        "correct": correct,
        "top1_accuracy": round(top1_accuracy, 6),
        "top_mistakes": top_mistakes,
        "per_class": per_class,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to:", OUTPUT_PATH)
    print("Correct:", correct)
    print("Total:", total)
    print("Top-1 Accuracy:", round(top1_accuracy, 6))


if __name__ == "__main__":
    main()