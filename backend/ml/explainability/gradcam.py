from pathlib import Path
from typing import List
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


class GradCAMExplainer:
    def __init__(self) -> None:
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)
        self.model_loader = ModelLoader(num_classes=len(self.class_names))
        self.transforms = get_inference_transforms(image_size=224)

    def _get_target_layer(self, model: torch.nn.Module, model_name: ModelName):
        if model_name == ModelName.EFFICIENTNET_B0:
            return model.features[-1]

        if model_name == ModelName.RESNET50:
            return model.layer4[-1]

        if model_name == ModelName.MOBILENET_V3_LARGE:
            return model.features[-1]

        raise ValueError(
            "Grad-CAM currently supports efficientnet_b0, resnet50, and mobilenet_v3_large only."
        )

    def explain(
        self,
        image: Image.Image,
        output_dir: str | Path,
        model_name: ModelName = ModelName.EFFICIENTNET_B0,
    ) -> dict:
        if model_name not in {
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        }:
            raise ValueError(
                "Grad-CAM currently supports efficientnet_b0, resnet50, and mobilenet_v3_large only."
            )

        loaded_model = self.model_loader.load_model(model_name)
        model = loaded_model.model
        device = loaded_model.device

        target_layer = self._get_target_layer(model, model_name)

        activations = []
        gradients = []

        def forward_hook(_module, _input, output):
            activations.append(output.detach())

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        resized_image = image.resize((224, 224)).convert("RGB")
        image_tensor = self.transforms(resized_image).unsqueeze(0).to(device)

        model.zero_grad()
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, pred_index].item())

        score = outputs[0, pred_index]
        score.backward()

        forward_handle.remove()
        backward_handle.remove()

        activation = activations[0]
        gradient = gradients[0]

        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        original_np = np.array(resized_image).astype(np.float32) / 255.0
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        overlay = 0.6 * original_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_suffix = uuid4().hex[:8]
        output_filename = f"gradcam_{model_name.value}_{pred_index}_{unique_suffix}.png"
        output_path = output_dir / output_filename

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(original_np)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            f"{model_name.value}\n{self.class_names[pred_index]} ({confidence:.4f})",
            fontsize=12,
        )
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return {
            "model_name": model_name.value,
            "predicted_class": self.class_names[pred_index],
            "confidence": round(confidence, 6),
            "heatmap_path": str(output_path),
            "heatmap_filename": output_filename,
        }