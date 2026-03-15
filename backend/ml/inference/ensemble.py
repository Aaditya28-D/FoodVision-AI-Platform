from time import perf_counter

import torch

from app.schemas.prediction import PredictionItem, PredictionResponse
from ml.inference.model_registry import ModelName


def run_effnet_resnet_ensemble(
    image,
    top_k: int,
    model_loader,
    transforms,
    class_names,
) -> PredictionResponse:
    start_time = perf_counter()

    eff_model = model_loader.load_model(ModelName.EFFICIENTNET_B0)
    res_model = model_loader.load_model(ModelName.RESNET50)

    image_tensor = transforms(image).unsqueeze(0).to(eff_model.device)

    with torch.no_grad():
        eff_logits = eff_model.model(image_tensor)
        res_logits = res_model.model(image_tensor)

        eff_probs = torch.softmax(eff_logits, dim=1)
        res_probs = torch.softmax(res_logits, dim=1)

        ensemble_probs = (eff_probs + res_probs) / 2.0

    inference_time_ms = (perf_counter() - start_time) * 1000

    top_probs, top_indices = torch.topk(ensemble_probs, k=top_k, dim=1)

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
        model_name="ensemble_efficientnet_b0_resnet50",
        top_k=top_k,
        predictions=predictions,
        inference_time_ms=round(inference_time_ms, 3),
        device=eff_model.device,
    )
