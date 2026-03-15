from collections import Counter
from typing import Dict, Optional

from app.schemas.prediction import PredictionResponse
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
