from pathlib import Path

from ultralytics import YOLO


class PartDetector:
    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
        conf: float = 0.25,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf

    def predict(self, image_path: str | Path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            end2end=False,
            verbose=False,
        )

        return results[0]
