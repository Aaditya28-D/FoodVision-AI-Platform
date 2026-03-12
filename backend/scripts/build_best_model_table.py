import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "models"

FILES = {
    "efficientnet_b0": BASE_DIR / "evaluation_efficientnet_b0.json",
    "resnet50": BASE_DIR / "evaluation_resnet50.json",
    "mobilenet_v3_large": BASE_DIR / "evaluation_mobilenet_v3_large.json",
}

def load_per_class(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["class_name"]: item["accuracy"] for item in data["per_class"]}

eff_map = load_per_class(FILES["efficientnet_b0"])
res_map = load_per_class(FILES["resnet50"])
mob_map = load_per_class(FILES["mobilenet_v3_large"])

all_foods = sorted(set(eff_map) | set(res_map) | set(mob_map))

rows = []
for food in all_foods:
    eff_acc = eff_map.get(food, 0.0)
    res_acc = res_map.get(food, 0.0)
    mob_acc = mob_map.get(food, 0.0)

    scores = {
        "efficientnet_b0": eff_acc,
        "resnet50": res_acc,
        "mobilenet_v3_large": mob_acc,
    }

    best_model = max(scores, key=scores.get)

    rows.append({
        "food": food,
        "efficientnet_b0_accuracy": round(eff_acc, 6),
        "resnet50_accuracy": round(res_acc, 6),
        "mobilenet_v3_large_accuracy": round(mob_acc, 6),
        "best_model": best_model,
    })

output_csv = BASE_DIR / "best_model_per_food.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "food",
            "efficientnet_b0_accuracy",
            "resnet50_accuracy",
            "mobilenet_v3_large_accuracy",
            "best_model",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved table to: {output_csv}")

for row in rows[:20]:
    print(row)