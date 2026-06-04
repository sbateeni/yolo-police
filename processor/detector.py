import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans

from config import (
    VEHICLE_MODEL_PATH, PLATE_MODEL_PATH,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    VEHICLE_TYPES, COLOR_NAME_THRESHOLD, COLOR_PALETTE,
)


class VehicleDetector:
    def __init__(self, model_path: str = VEHICLE_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = VEHICLE_TYPES.get(cls_id, "unknown")
            vehicle_crop = frame[y1:y2, x1:x2]
            color = self._get_dominant_color(vehicle_crop)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "type": label,
                "color": color,
                "class_id": cls_id,
            })
        return detections

    def _get_dominant_color(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"
        pixels = crop.reshape(-1, 3)
        if len(pixels) < 10:
            return "unknown"
        kmeans = KMeans(n_clusters=1, n_init=1, random_state=0)
        kmeans.fit(pixels)
        dominant = kmeans.cluster_centers_[0]
        return self._closest_color(dominant)

    def _closest_color(self, rgb: np.ndarray) -> str:
        best_name = "unknown"
        best_dist = float("inf")
        for name, ref in COLOR_PALETTE.items():
            dist = np.linalg.norm(rgb - np.array(ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name if best_dist < COLOR_NAME_THRESHOLD else "unknown"


class PlateDetector:
    def __init__(self, model_path: str = PLATE_MODEL_PATH):
        candidate = Path(model_path)
        if not candidate.exists():
            fallback = Path(VEHICLE_MODEL_PATH)
            if fallback.exists():
                model_path = str(fallback)
            else:
                raise FileNotFoundError(
                    f"Plate model not found: {model_path} and fallback missing: {fallback}"
                )
        self.model = YOLO(str(model_path))

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            plate_crop = frame[y1:y2, x1:x2]
            plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "crop": plate_crop,
            })
        return plates

    def get_plate_color(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"
        pixels = crop.reshape(-1, 3)
        if len(pixels) < 10:
            return "unknown"
        kmeans = KMeans(n_clusters=2, n_init=1, random_state=0)
        kmeans.fit(pixels)
        bg_idx = np.argmax(np.bincount(kmeans.labels_))
        bg_color = kmeans.cluster_centers_[bg_idx]
        return self._closest_color_name(bg_color)

    def _closest_color_name(self, rgb: np.ndarray) -> str:
        best_name = "unknown"
        best_dist = float("inf")
        for name, ref in COLOR_PALETTE.items():
            dist = np.linalg.norm(rgb - np.array(ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name if best_dist < COLOR_NAME_THRESHOLD else "unknown"
