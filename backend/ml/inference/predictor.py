from collections import Counter
from time import perf_counter
from typing import Dict, List, Optional

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
    HIGH_CONF_THRESHOLD = 0.75
    CONF_MARGIN = 0.10
    SPECIALIST_MIN_CONFIDENCE = 0.55
    DEFAULT_FALLBACK_MODEL = ModelName.EFFICIENTNET_B0

    SPECIALIST_CLASS_MODEL_MAP = {
        "apple_pie": ModelName.EFFICIENTNET_B0,
        "baby_back_ribs": ModelName.EFFICIENTNET_B0,
        "baklava": ModelName.RESNET50,
        "beef_carpaccio": ModelName.MOBILENET_V3_LARGE,
        "beef_tartare": ModelName.RESNET50,
        "beet_salad": ModelName.MOBILENET_V3_LARGE,
        "beignets": ModelName.EFFICIENTNET_B0,
        "bibimbap": ModelName.RESNET50,
        "bread_pudding": ModelName.EFFICIENTNET_B0,
        "breakfast_burrito": ModelName.RESNET50,
        "bruschetta": ModelName.RESNET50,
        "caesar_salad": ModelName.EFFICIENTNET_B0,
        "cannoli": ModelName.RESNET50,
        "caprese_salad": ModelName.MOBILENET_V3_LARGE,
        "carrot_cake": ModelName.RESNET50,
        "ceviche": ModelName.EFFICIENTNET_B0,
        "cheese_plate": ModelName.MOBILENET_V3_LARGE,
        "cheesecake": ModelName.MOBILENET_V3_LARGE,
        "chicken_curry": ModelName.RESNET50,
        "chicken_quesadilla": ModelName.RESNET50,
        "chicken_wings": ModelName.MOBILENET_V3_LARGE,
        "chocolate_cake": ModelName.RESNET50,
        "chocolate_mousse": ModelName.EFFICIENTNET_B0,
        "churros": ModelName.MOBILENET_V3_LARGE,
        "clam_chowder": ModelName.EFFICIENTNET_B0,
        "club_sandwich": ModelName.EFFICIENTNET_B0,
        "crab_cakes": ModelName.RESNET50,
        "creme_brulee": ModelName.MOBILENET_V3_LARGE,
        "croque_madame": ModelName.EFFICIENTNET_B0,
        "cup_cakes": ModelName.EFFICIENTNET_B0,
        "deviled_eggs": ModelName.EFFICIENTNET_B0,
        "donuts": ModelName.EFFICIENTNET_B0,
        "dumplings": ModelName.EFFICIENTNET_B0,
        "edamame": ModelName.EFFICIENTNET_B0,
        "eggs_benedict": ModelName.EFFICIENTNET_B0,
        "escargots": ModelName.EFFICIENTNET_B0,
        "falafel": ModelName.RESNET50,
        "filet_mignon": ModelName.RESNET50,
        "fish_and_chips": ModelName.EFFICIENTNET_B0,
        "foie_gras": ModelName.EFFICIENTNET_B0,
        "french_fries": ModelName.EFFICIENTNET_B0,
        "french_onion_soup": ModelName.EFFICIENTNET_B0,
        "french_toast": ModelName.EFFICIENTNET_B0,
        "fried_calamari": ModelName.RESNET50,
        "fried_rice": ModelName.RESNET50,
        "frozen_yogurt": ModelName.EFFICIENTNET_B0,
        "garlic_bread": ModelName.RESNET50,
        "gnocchi": ModelName.RESNET50,
        "greek_salad": ModelName.EFFICIENTNET_B0,
        "grilled_cheese_sandwich": ModelName.EFFICIENTNET_B0,
        "grilled_salmon": ModelName.EFFICIENTNET_B0,
        "guacamole": ModelName.EFFICIENTNET_B0,
        "gyoza": ModelName.MOBILENET_V3_LARGE,
        "hamburger": ModelName.EFFICIENTNET_B0,
        "hot_and_sour_soup": ModelName.EFFICIENTNET_B0,
        "hot_dog": ModelName.EFFICIENTNET_B0,
        "huevos_rancheros": ModelName.EFFICIENTNET_B0,
        "hummus": ModelName.RESNET50,
        "ice_cream": ModelName.EFFICIENTNET_B0,
        "lasagna": ModelName.EFFICIENTNET_B0,
        "lobster_bisque": ModelName.MOBILENET_V3_LARGE,
        "lobster_roll_sandwich": ModelName.RESNET50,
        "macaroni_and_cheese": ModelName.EFFICIENTNET_B0,
        "macarons": ModelName.EFFICIENTNET_B0,
        "miso_soup": ModelName.EFFICIENTNET_B0,
        "mussels": ModelName.MOBILENET_V3_LARGE,
        "nachos": ModelName.MOBILENET_V3_LARGE,
        "omelette": ModelName.EFFICIENTNET_B0,
        "onion_rings": ModelName.EFFICIENTNET_B0,
        "oysters": ModelName.RESNET50,
        "pad_thai": ModelName.EFFICIENTNET_B0,
        "paella": ModelName.MOBILENET_V3_LARGE,
        "pancakes": ModelName.EFFICIENTNET_B0,
        "panna_cotta": ModelName.EFFICIENTNET_B0,
        "peking_duck": ModelName.MOBILENET_V3_LARGE,
        "pho": ModelName.RESNET50,
        "pizza": ModelName.EFFICIENTNET_B0,
        "pork_chop": ModelName.EFFICIENTNET_B0,
        "poutine": ModelName.EFFICIENTNET_B0,
        "prime_rib": ModelName.EFFICIENTNET_B0,
        "pulled_pork_sandwich": ModelName.EFFICIENTNET_B0,
        "ramen": ModelName.EFFICIENTNET_B0,
        "ravioli": ModelName.RESNET50,
        "red_velvet_cake": ModelName.EFFICIENTNET_B0,
        "risotto": ModelName.RESNET50,
        "samosa": ModelName.RESNET50,
        "sashimi": ModelName.RESNET50,
        "scallops": ModelName.EFFICIENTNET_B0,
        "seaweed_salad": ModelName.MOBILENET_V3_LARGE,
        "shrimp_and_grits": ModelName.EFFICIENTNET_B0,
        "spaghetti_bolognese": ModelName.RESNET50,
        "spaghetti_carbonara": ModelName.EFFICIENTNET_B0,
        "spring_rolls": ModelName.EFFICIENTNET_B0,
        "steak": ModelName.EFFICIENTNET_B0,
        "strawberry_shortcake": ModelName.RESNET50,
        "sushi": ModelName.EFFICIENTNET_B0,
        "takoyaki": ModelName.RESNET50,
        "tacos": ModelName.EFFICIENTNET_B0,
        "tiramisu": ModelName.EFFICIENTNET_B0,
        "tuna_tartare": ModelName.RESNET50,
        "waffles": ModelName.EFFICIENTNET_B0,
    }

    def __init__(self) -> None:
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)
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

        inference_time_ms = (perf_counter() - start_time) * 1000

        return self._build_prediction_response(
            model_name="ensemble_efficientnet_b0_resnet50",
            probs=ensemble_probs,
            top_k=top_k,
            inference_time_ms=inference_time_ms,
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

    def _majority_vote(
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

    def _class_specialist_winner(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> Optional[PredictionResponse]:
        specialist_candidates: List[PredictionResponse] = []

        for model_name, response in model_outputs.items():
            top_pred = response.predictions[0]
            predicted_class = top_pred.class_name
            confidence = float(top_pred.confidence)

            specialist_model = self.SPECIALIST_CLASS_MODEL_MAP.get(predicted_class)

            if specialist_model == model_name and confidence >= self.SPECIALIST_MIN_CONFIDENCE:
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

    def _confidence_winner(
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

        if best_conf >= self.HIGH_CONF_THRESHOLD and (best_conf - second_conf) >= self.CONF_MARGIN:
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

    def _fallback_winner(
        self,
        model_outputs: Dict[ModelName, PredictionResponse],
    ) -> PredictionResponse:
        selected = model_outputs[self.DEFAULT_FALLBACK_MODEL]
        return PredictionResponse(
            model_name=f"smart_router->fallback->{self.DEFAULT_FALLBACK_MODEL.value}",
            top_k=selected.top_k,
            predictions=selected.predictions,
            inference_time_ms=round(
                sum(resp.inference_time_ms for resp in model_outputs.values()),
                3,
            ),
            device=selected.device,
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

        majority_result = self._majority_vote(model_outputs)
        if majority_result is not None:
            return majority_result

        specialist_result = self._class_specialist_winner(model_outputs)
        if specialist_result is not None:
            return specialist_result

        confidence_result = self._confidence_winner(model_outputs)
        if confidence_result is not None:
            return confidence_result

        return self._fallback_winner(model_outputs)

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