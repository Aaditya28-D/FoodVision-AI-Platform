from pathlib import Path
from typing import List

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

    def explain(
        self,
        image: Image.Image,
        output_dir: str | Path,
        model_name: ModelName = ModelName.EFFICIENTNET_B0,
    ) -> dict:
        if model_name != ModelName.EFFICIENTNET_B0:
            raise ValueError("Grad-CAM currently supports efficientnet_b0 only.")

        loaded_model = self.model_loader.load_model(model_name)
        model = loaded_model.model
        device = loaded_model.device

        target_layer = model.features[-1]

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

        activation = activations[0]          # shape: [1, C, H, W]
        gradient = gradients[0]              # shape: [1, C, H, W]

        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1, keepdim=True)  # [1,1,H,W]
        cam = F.relu(cam)

        # Resize CAM to image size
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

        output_path = output_dir / f"gradcam_{model_name.value}_{pred_index}.png"
        plt.imsave(output_path, overlay)

        return {
            "model_name": model_name.value,
            "predicted_class": self.class_names[pred_index],
            "confidence": round(confidence, 6),
            "heatmap_path": str(output_path),
        }