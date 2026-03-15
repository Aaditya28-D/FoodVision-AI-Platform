from time import perf_counter

import torch

from app.schemas.prediction import PredictionItem, PredictionResponse
from ml.inference.model_registry import ModelName


def _build_ensemble_response(
    probs: torch.Tensor,
    top_k: int,
    inference_time_ms: float,
    device: str,
    class_names: list[str],
    model_name: str,
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

    return _build_ensemble_response(
        probs=ensemble_probs,
        top_k=top_k,
        inference_time_ms=inference_time_ms,
        device=eff_model.device,
        class_names=class_names,
        model_name="ensemble_efficientnet_b0_resnet50",
    )


def run_effnet_resnet_weighted_ensemble(
    image,
    top_k: int,
    model_loader,
    transforms,
    class_names,
    eff_weight: float = 0.6,
    res_weight: float = 0.4,
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

        ensemble_probs = (eff_weight * eff_probs) + (res_weight * res_probs)

    inference_time_ms = (perf_counter() - start_time) * 1000

    return _build_ensemble_response(
        probs=ensemble_probs,
        top_k=top_k,
        inference_time_ms=inference_time_ms,
        device=eff_model.device,
        class_names=class_names,
        model_name=f"ensemble_efficientnet_b0_resnet50_weighted_{eff_weight:.1f}_{res_weight:.1f}",
    )


def run_effnet_mobilenet_ensemble(
    image,
    top_k: int,
    model_loader,
    transforms,
    class_names,
) -> PredictionResponse:
    start_time = perf_counter()

    eff_model = model_loader.load_model(ModelName.EFFICIENTNET_B0)
    mob_model = model_loader.load_model(ModelName.MOBILENET_V3_LARGE)

    image_tensor = transforms(image).unsqueeze(0).to(eff_model.device)

    with torch.no_grad():
        eff_logits = eff_model.model(image_tensor)
        mob_logits = mob_model.model(image_tensor)

        eff_probs = torch.softmax(eff_logits, dim=1)
        mob_probs = torch.softmax(mob_logits, dim=1)

        ensemble_probs = (eff_probs + mob_probs) / 2.0

    inference_time_ms = (perf_counter() - start_time) * 1000

    return _build_ensemble_response(
        probs=ensemble_probs,
        top_k=top_k,
        inference_time_ms=inference_time_ms,
        device=eff_model.device,
        class_names=class_names,
        model_name="ensemble_efficientnet_b0_mobilenet_v3_large",
    )


def run_effnet_mobilenet_weighted_ensemble(
    image,
    top_k: int,
    model_loader,
    transforms,
    class_names,
    eff_weight: float = 0.6,
    mob_weight: float = 0.4,
) -> PredictionResponse:
    start_time = perf_counter()

    eff_model = model_loader.load_model(ModelName.EFFICIENTNET_B0)
    mob_model = model_loader.load_model(ModelName.MOBILENET_V3_LARGE)

    image_tensor = transforms(image).unsqueeze(0).to(eff_model.device)

    with torch.no_grad():
        eff_logits = eff_model.model(image_tensor)
        mob_logits = mob_model.model(image_tensor)

        eff_probs = torch.softmax(eff_logits, dim=1)
        mob_probs = torch.softmax(mob_logits, dim=1)

        ensemble_probs = (eff_weight * eff_probs) + (mob_weight * mob_probs)

    inference_time_ms = (perf_counter() - start_time) * 1000

    return _build_ensemble_response(
        probs=ensemble_probs,
        top_k=top_k,
        inference_time_ms=inference_time_ms,
        device=eff_model.device,
        class_names=class_names,
        model_name=f"ensemble_efficientnet_b0_mobilenet_v3_large_weighted_{eff_weight:.1f}_{mob_weight:.1f}",
    )
