from pydantic import BaseModel


class BulkClassificationSummary(BaseModel):
    total_files: int
    classified_files: int
    low_confidence_files: int
    strategy: str
    confidence_threshold: float
    zip_filename: str
    download_url: str
