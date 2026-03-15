from collections import Counter
from time import perf_counter
from typing import Dict, Optional

import torch

from app.schemas.prediction import PredictionItem, PredictionResponse
from ml.inference.model_registry import ModelName
from ml.inference.specialist_map import SPECIALIST_CLASS_MODEL_MAP


class SmartRouter:
    def __init__(
        self,
        high_conf_threshold: float = 0.75,
        conf_margin: float = 0.10,
        specialist_min_confidence: float = 0.55,
        default_fallback_model: ModelName = ModelName.EFFICIENTNET_B0,
    ) -> None:
        self.high_conf_threshold = high_conf_threshold
        self.conf_margin = conf_margin
        self.specialist_min_confidence = specialist_min_confidence
        self.default_fallback_model = default_fallback_model

    def majority_vote(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> Optional[PredictionResponse]:
        top_classes = [resp.predictions[0].class_name for resp in model_outputs.values()]
        counts = Counter(top_classes)
        majority_class, count = counts.most_common(1)[0]

        if count < 2:
            return None

        agreeing = [
            resp for resp in model_outputs.values()
            if resp.predictions[0].class_name == majority_class
        ]

        best_agreeing = max(
            agreeing,
            key=lambda resp: float(resp.predictions[0].confidence),
        )

        return PredictionResponse(
            model_name=f"smart_router->majority_vote->{best_agreeing.model_name}",
            top_k=best_agreeing.top_k,
            predictions=best_agreeing.predictions,
            inference_time_ms=round(
                sum(resp.inference_time_ms for resp in model_outputs.values()),
                3,
            ),
            device=best_agreeing.device,
        )

    def class_specialist_winner(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> Optional[PredictionResponse]:
        specialist_candidates: list[PredictionResponse] = []

        for model_name, response in model_outputs.items():
            top_pred = response.predictions[0]
            predicted_class = top_pred.class_name
            confidence = float(top_pred.confidence)

            specialist_model = SPECIALIST_CLASS_MODEL_MAP.get(predicted_class)

            if specialist_model == model_name and confidence >= self.specialist_min_confidence:
                specialist_candidates.append(response)

        if not specialist_candidates:
            return None

        best_specialist = max(
            specialist_candidates,
            key=lambda resp: float(resp.predictions[0].confidence),
        )

        return PredictionResponse(
            model_name=f"smart_router->class_specialist->{best_specialist.model_name}",
            top_k=best_specialist.top_k,
            predictions=best_specialist.predictions,
            inference_time_ms=round(
                sum(resp.inference_time_ms for resp in model_outputs.values()),
                3,
            ),
            device=best_specialist.device,
        )

    def confidence_winner(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> Optional[PredictionResponse]:
        ranked = sorted(
            model_outputs.values(),
            key=lambda resp: float(resp.predictions[0].confidence),
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1]

        best_conf = float(best.predictions[0].confidence)
        second_conf = float(second.predictions[0].confidence)

        if best_conf >= self.high_conf_threshold and (best_conf - second_conf) >= self.conf_margin:
            return PredictionResponse(
                model_name=f"smart_router->confidence_winner->{best.model_name}",
                top_k=best.top_k,
                predictions=best.predictions,
                inference_time_ms=round(
                    sum(resp.inference_time_ms for resp in model_outputs.values()),
                    3,
                ),
                device=best.device,
            )

        return None

    def fallback_winner(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> PredictionResponse:
        selected = model_outputs[self.default_fallback_model]

        return PredictionResponse(
            model_name=f"smart_router->fallback->{self.default_fallback_model.value}",
            top_k=selected.top_k,
            predictions=selected.predictions,
            inference_time_ms=round(
                sum(resp.inference_time_ms for resp in model_outputs.values()),
                3,
            ),
            device=selected.device,
        )


def _build_prediction_response(
    model_name: str,
    probs: torch.Tensor,
    top_k: int,
    inference_time_ms: float,
    device: str,
    class_names: list[str],
) -> PredictionResponse:
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_probs = top_probs.squeeze(0).tolist()
    top_indices = top_indices.squeeze(0).tolist()

    predictions = [
        PredictionItem(
            class_name=class_names[idx],
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


def _run_single_model_for_router(
    image,
    model_name: ModelName,
    top_k: int,
    model_loader,
    transforms,
    class_names: list[str],
) -> PredictionResponse:
    start_time = perf_counter()

    loaded_model = model_loader.load_model(model_name)
    image_tensor = transforms(image).unsqueeze(0).to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    inference_time_ms = (perf_counter() - start_time) * 1000

    return _build_prediction_response(
        model_name=loaded_model.model_name,
        probs=probabilities,
        top_k=top_k,
        inference_time_ms=inference_time_ms,
        device=loaded_model.device,
        class_names=class_names,
    )


def run_smart_router(
    image,
    top_k: int,
    model_loader,
    transforms,
    class_names: list[str],
) -> PredictionResponse:
    model_outputs: Dict[ModelName, PredictionResponse] = {
        ModelName.EFFICIENTNET_B0: _run_single_model_for_router(
            image=image,
            model_name=ModelName.EFFICIENTNET_B0,
            top_k=top_k,
            model_loader=model_loader,
            transforms=transforms,
            class_names=class_names,
        ),
        ModelName.RESNET50: _run_single_model_for_router(
            image=image,
            model_name=ModelName.RESNET50,
            top_k=top_k,
            model_loader=model_loader,
            transforms=transforms,
            class_names=class_names,
        ),
        ModelName.MOBILENET_V3_LARGE: _run_single_model_for_router(
            image=image,
            model_name=ModelName.MOBILENET_V3_LARGE,
            top_k=top_k,
            model_loader=model_loader,
            transforms=transforms,
            class_names=class_names,
        ),
    }

    router = SmartRouter()

    majority_result = router.majority_vote(model_outputs)
    if majority_result is not None:
        return majority_result

    specialist_result = router.class_specialist_winner(model_outputs)
    if specialist_result is not None:
        return specialist_result

    confidence_result = router.confidence_winner(model_outputs)
    if confidence_result is not None:
        return confidence_result

    return router.fallback_winner(model_outputs)
