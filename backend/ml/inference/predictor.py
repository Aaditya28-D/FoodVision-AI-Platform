from time import perf_counter

import torch

from app.core.config import settings
from app.schemas.prediction import ComparisonResponse, PredictionItem, PredictionResponse
from ml.inference.class_names import load_class_names
from ml.inference.comparison import build_comparison_response
from ml.inference.ensemble import run_effnet_resnet_ensemble
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.routers import run_smart_router
from ml.inference.transforms import get_inference_transforms


class FoodPredictor:
    def __init__(self) -> None:
        self.class_names = load_class_names(settings.CLASS_NAMES_PATH)
        self.model_loader = ModelLoader(num_classes=len(self.class_names))
        self.transforms = get_inference_transforms(image_size=224)

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
        image,
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
        image,
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
        image,
        top_k: int = 5,
    ) -> PredictionResponse:
        return run_effnet_resnet_ensemble(
            image=image,
            top_k=top_k,
            model_loader=self.model_loader,
            transforms=self.transforms,
            class_names=self.class_names,
        )

    def predict_smart(
        self,
        image,
        top_k: int = 5,
    ) -> PredictionResponse:
        return run_smart_router(
            image=image,
            top_k=top_k,
            model_loader=self.model_loader,
            transforms=self.transforms,
            class_names=self.class_names,
        )

    def compare_models(
        self,
        image,
        top_k: int = 5,
    ) -> ComparisonResponse:
        return self.compare_specific_models(
            image=image,
            model_names=[
                ModelName.EFFICIENTNET_B0,
                ModelName.RESNET50,
                ModelName.MOBILENET_V3_LARGE,
            ],
            top_k=top_k,
        )

    def compare_specific_models(
        self,
        image,
        model_names: list[ModelName],
        top_k: int = 5,
    ) -> ComparisonResponse:
        responses = [
            self.predict(image=image, model_name=model_name, top_k=top_k)
            for model_name in model_names
        ]
        return build_comparison_response(responses=responses, top_k=top_k)
