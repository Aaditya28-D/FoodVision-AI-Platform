from time import perf_counter
from typing import Dict, List

import torch
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import (
    ComparisonResult,
    PredictionItem,
    PredictionResponse,
)
from ml.inference.class_names import load_class_names
from ml.inference.comparison import build_comparison_response
from ml.inference.ensemble import EnsemblePredictor
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.routers import SmartRouter
from ml.inference.transforms import get_inference_transforms


class FoodPredictor:
    def __init__(self) -> None:
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)
        self.model_loader = ModelLoader(num_classes=len(self.class_names))
        self.transforms = get_inference_transforms(image_size=224)

        self.ensemble_predictor = EnsemblePredictor(
            model_loader=self.model_loader,
            class_names=self.class_names,
        )
        self.smart_router = SmartRouter(
            high_conf_threshold=0.75,
            conf_margin=0.10,
            specialist_min_confidence=0.55,
            default_fallback_model=ModelName.EFFICIENTNET_B0,
        )

    def _build_prediction_response(
        self,
        model_name: str,
        probs: torch.Tensor,
        top_k: int,
        inference_time_ms: float,
        device: str,
    ) -> PredictionResponse:
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        predictions = [
            PredictionItem(
                class_name=self.class_names[idx],
                confidence=round(float(prob), 6),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        return PredictionResponse(
            model_name=model_name,
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
            device=device,
        )

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

        inference_time_ms = (perf_counter() - start_time) * 1000

        return self._build_prediction_response(
            model_name=loaded_model.model_name,
            probs=probabilities,
            top_k=top_k,
            inference_time_ms=inference_time_ms,
            device=loaded_model.device,
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
        return self.ensemble_predictor.predict(
            image=image,
            top_k=top_k,
        )

    def predict_smart(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> PredictionResponse:
        model_outputs: Dict[ModelName, PredictionResponse] = {
            ModelName.EFFICIENTNET_B0: self.predict(
                image=image,
                model_name=ModelName.EFFICIENTNET_B0,
                top_k=top_k,
            ),
            ModelName.RESNET50: self.predict(
                image=image,
                model_name=ModelName.RESNET50,
                top_k=top_k,
            ),
            ModelName.MOBILENET_V3_LARGE: self.predict(
                image=image,
                model_name=ModelName.MOBILENET_V3_LARGE,
                top_k=top_k,
            ),
        }

        majority_result = self.smart_router.majority_vote(model_outputs)
        if majority_result is not None:
            return majority_result

        specialist_result = self.smart_router.class_specialist_winner(model_outputs)
        if specialist_result is not None:
            return specialist_result

        confidence_result = self.smart_router.confidence_winner(model_outputs)
        if confidence_result is not None:
            return confidence_result

        return self.smart_router.fallback_winner(model_outputs)

    def compare_models(
        self,
        image: Image.Image,
        top_k: int = 5,
    ):
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

        return build_comparison_response(
            results=results,
            top_k=top_k,
        )
