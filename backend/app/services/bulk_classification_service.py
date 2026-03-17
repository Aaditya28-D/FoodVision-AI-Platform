import csv
import shutil
import uuid
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

from app.core.config import settings
from app.services.prediction_service import PredictionService


class BulkClassificationService:
    def __init__(self) -> None:
        self.prediction_service = PredictionService()
        self.output_root = settings.ARTIFACTS_DIR / "bulk_classification"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _safe_open_image(self, file_bytes: bytes) -> Image.Image:
        return Image.open(__import__("io").BytesIO(file_bytes)).convert("RGB")

    def classify_files(
        self,
        files: list[tuple[str, bytes]],
        strategy: str,
        confidence_threshold: float,
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.output_root / job_id
        grouped_dir = job_dir / "classified_output"
        grouped_dir.mkdir(parents=True, exist_ok=True)

        results_csv_path = grouped_dir / "results.csv"
        zip_filename = f"bulk_classification_{job_id}.zip"
        zip_path = job_dir / zip_filename

        classified_count = 0
        low_confidence_count = 0
        rows: list[dict] = []

        for original_name, file_bytes in files:
            try:
                image = self._safe_open_image(file_bytes)
            except Exception:
                low_dir = grouped_dir / "invalid_image"
                low_dir.mkdir(parents=True, exist_ok=True)
                output_path = low_dir / original_name
                output_path.write_bytes(file_bytes)

                rows.append(
                    {
                        "filename": original_name,
                        "predicted_class": "invalid_image",
                        "confidence": "",
                        "strategy": strategy,
                        "status": "invalid_image",
                    }
                )
                continue

            response = self.prediction_service.predict(
                image=image,
                strategy=strategy,
                top_k=1,
            )

            top_prediction = response.predictions[0]
            predicted_class = top_prediction.class_name
            confidence = float(top_prediction.confidence)

            if confidence >= confidence_threshold:
                target_folder = predicted_class
                classified_count += 1
                status = "classified"
            else:
                target_folder = "low_confidence"
                low_confidence_count += 1
                status = "low_confidence"

            target_dir = grouped_dir / target_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            output_path = target_dir / original_name
            output_path.write_bytes(file_bytes)

            rows.append(
                {
                    "filename": original_name,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.6f}",
                    "strategy": strategy,
                    "status": status,
                }
            )

        with results_csv_path.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["filename", "predicted_class", "confidence", "strategy", "status"],
            )
            writer.writeheader()
            writer.writerows(rows)

        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
            for path in grouped_dir.rglob("*"):
                if path.is_file():
                    arcname = path.relative_to(job_dir)
                    zip_file.write(path, arcname=arcname)

        return {
            "total_files": len(files),
            "classified_files": classified_count,
            "low_confidence_files": low_confidence_count,
            "strategy": strategy,
            "confidence_threshold": confidence_threshold,
            "zip_filename": zip_filename,
            "download_url": f"/artifacts/bulk_classification/{job_id}/{zip_filename}",
        }
