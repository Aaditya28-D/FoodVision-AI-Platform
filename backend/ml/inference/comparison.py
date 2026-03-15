from collections import Counter

from app.schemas.prediction import (
    BattleSummary,
    ComparisonResponse,
    ComparisonResult,
    PredictionResponse,
)


def build_comparison_response(
    responses: list[PredictionResponse],
    top_k: int,
) -> ComparisonResponse:
    results = [
        ComparisonResult(
            model_name=response.model_name,
            predictions=response.predictions,
            inference_time_ms=response.inference_time_ms,
            device=response.device or "cpu",
            top_prediction=response.predictions[0],
        )
        for response in responses
    ]

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
