import cv2
import numpy as np
import easyocr
import re

from config import OCR_LANGUAGES, OCR_GPU


class PlateReader:
    def __init__(self, languages: list[str] = None, gpu: bool = OCR_GPU):
        self.reader = easyocr.Reader(languages or OCR_LANGUAGES, gpu=gpu)

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        results = self.reader.readtext(enhanced, detail=1)
        if not results:
            return "", 0.0

        # best result by confidence
        text, confidence = "", 0.0
        for bbox, label, score in results:
            cleaned = self._normalize_plate_text(label)
            if cleaned and score > confidence:
                text = cleaned
                confidence = float(score)

        return text, confidence

    def read_raw(self, crop: np.ndarray) -> list[dict]:
        if crop is None or crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = self.reader.readtext(enhanced, detail=1)
        return [
            {"bbox": r[0], "text": self._normalize_plate_text(r[1].strip()), "confidence": round(r[2], 4)}
            for r in results
        ]

    def _normalize_plate_text(self, text: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9- ]+", "", text)
        normalized = normalized.strip().replace(" ", "")
        return normalized
