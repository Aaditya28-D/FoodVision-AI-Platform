from time import perf_counter

import torch
from PIL import Image

from app.schemas.prediction import PredictionResponse
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


class EnsemblePredictor:
    def __init__(self, model_loader: ModelLoader, class_names: list[str]) -> None:
        self.model_loader = model_loader
        self.class_names = class_names
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

        from app.schemas.prediction import PredictionItem

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

    def predict(
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

        inference_time_ms = (perf_counter() - start_time) * 1000

        return self._build_prediction_response(
            model_name="ensemble_efficientnet_b0_resnet50",
            probs=ensemble_probs,
            top_k=top_k,
            inference_time_ms=inference_time_ms,
            device=eff_model.device,
        )
