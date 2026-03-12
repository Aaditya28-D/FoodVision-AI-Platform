from collections import Counter
from time import perf_counter
from typing import List

import torch
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import (
    BattleSummary,
    ComparisonResponse,
    ComparisonResult,
    PredictionItem,
    PredictionResponse,
)
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


class FoodPredictor:
    def __init__(self) -> None:
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)
        self.model_loader = ModelLoader(num_classes=len(self.class_names))
        self.transforms = get_inference_transforms(image_size=224)

    def _run_single_model(
        self,
        image: Image.Image,
        model_name: ModelName,
        top_k: int = 5,
    ) -> PredictionResponse:
        start_time = perf_counter()

        loaded_model = self.model_loader.load_model(model_name)
        image_tensor = self.transforms(image).unsqueeze(0).to(loaded_model.device)

        with torch.no_grad():
            outputs = loaded_model.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        predictions = [
            PredictionItem(
                class_name=self.class_names[idx],
                confidence=round(float(prob), 6),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        inference_time_ms = (perf_counter() - start_time) * 1000

        return PredictionResponse(
            model_name=loaded_model.model_name,
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
            device=loaded_model.device,
        )

    def _run_ensemble(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> PredictionResponse:
        start_time = perf_counter()

        eff_model = self.model_loader.load_model(ModelName.EFFICIENTNET_B0)
        res_model = self.model_loader.load_model(ModelName.RESNET50)

        image_tensor = self.transforms(image).unsqueeze(0).to(eff_model.device)

        with torch.no_grad():
            eff_logits = eff_model.model(image_tensor)
            res_logits = res_model.model(image_tensor)

            eff_probs = torch.softmax(eff_logits, dim=1)
            res_probs = torch.softmax(res_logits, dim=1)

            ensemble_probs = (eff_probs + res_probs) / 2.0
            top_probs, top_indices = torch.topk(ensemble_probs, k=top_k, dim=1)

        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        predictions = [
            PredictionItem(
                class_name=self.class_names[idx],
                confidence=round(float(prob), 6),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        inference_time_ms = (perf_counter() - start_time) * 1000

        return PredictionResponse(
            model_name="ensemble_efficientnet_b0_resnet50",
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
            device=eff_model.device,
        )

    def predict(
        self,
        image: Image.Image,
        model_name: ModelName = ModelName.MOBILENET_V3_LARGE,
        top_k: int = 5,
    ) -> PredictionResponse:
        return self._run_single_model(
            image=image,
            model_name=model_name,
            top_k=top_k,
        )

    def predict_ensemble(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> PredictionResponse:
        return self._run_ensemble(
            image=image,
            top_k=top_k,
        )

    def compare_models(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> ComparisonResponse:
        comparison_models = [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ]

        results: List[ComparisonResult] = []

        for model_name in comparison_models:
            response = self._run_single_model(
                image=image,
                model_name=model_name,
                top_k=top_k,
            )

            top_prediction = response.predictions[0]

            results.append(
                ComparisonResult(
                    model_name=response.model_name,
                    predictions=response.predictions,
                    inference_time_ms=response.inference_time_ms,
                    device=response.device or "cpu",
                    top_prediction=top_prediction,
                )
            )

        fastest_result = min(results, key=lambda item: item.inference_time_ms)
        highest_confidence_result = max(
            results,
            key=lambda item: item.top_prediction.confidence,
        )

        top_labels = [result.top_prediction.class_name for result in results]
        label_counts = Counter(top_labels)
        majority_label = label_counts.most_common(1)[0][0]
        all_models_agree = len(set(top_labels)) == 1

        summary = BattleSummary(
            fastest_model=fastest_result.model_name,
            highest_confidence_model=highest_confidence_result.model_name,
            all_models_agree=all_models_agree,
            majority_label=majority_label,
        )

        return ComparisonResponse(
            top_k=top_k,
            results=results,
            summary=summary,
        )