import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import (
    VEHICLE_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
)


class PersonDetector:
    """Detect persons and extract faces using YOLOv8."""
    
    def __init__(self, model_path: str = VEHICLE_MODEL_PATH, gender_model_path: str = None):
        if gender_model_path is None:
            from config import GENDER_MODEL_PATH
            gender_model_path = GENDER_MODEL_PATH
            
        # Use default YOLO model; detects person class (ID 0)
        self.model = YOLO(model_path)
        
        # Load the YOLO classification model for gender
        import os
        if os.path.exists(gender_model_path):
            self.gender_model = YOLO(gender_model_path)
        else:
            self.gender_model = None
            
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Detect person objects in frame."""
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        persons = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # Only person class
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            person_crop = frame[y1:y2, x1:x2]
            persons.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "crop": person_crop,
                "class_id": cls_id,
            })
        return persons
    
    def extract_faces(self, person_crop: np.ndarray) -> list[dict]:
        """Extract faces from person crop."""
        if person_crop is None or person_crop.size == 0:
            return []
        
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        face_list = []
        for (x, y, w, h) in faces:
            face_crop = person_crop[y:y+h, x:x+w]
            face_list.append({
                "bbox": (x, y, w, h),
                "crop": face_crop,
            })
        return face_list
    
    def estimate_gender_clothing(self, person_crop: np.ndarray, face_crop: np.ndarray = None) -> dict:
        """
        Estimate gender and clothing.
        Gender uses YOLO classification model if available, otherwise heuristics.
        """
        if person_crop is None or person_crop.size == 0:
            return {"gender": "unknown", "clothing": "unknown"}
        
        h, w = person_crop.shape[:2]
        upper_half = person_crop[:h//2, :]
        lower_half = person_crop[h//2:, :]
        
        # Simple color analysis for clothing
        upper_avg = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV).mean(axis=(0, 1))
        lower_avg = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV).mean(axis=(0, 1))
        
        # Heuristic: if upper part is dark, likely wearing dark clothing
        upper_brightness = upper_avg[2]
        lower_brightness = lower_avg[2]
        
        clothing = "dark" if upper_brightness < 100 else "light"
        if upper_brightness > 150:
            clothing = "bright"
        
        gender = "unknown"
        if self.gender_model is not None:
            try:
                # Use face_crop if available, otherwise person_crop
                target_crop = face_crop if (face_crop is not None and face_crop.size > 0) else person_crop
                res = self.gender_model(target_crop, verbose=False)[0]
                top_class = int(res.probs.top1)
                gender = res.names[top_class]  # 'male' or 'female'
            except Exception as e:
                pass
                
        # Fallback heuristic if model fails
        if gender not in ["male", "female"]:
            aspect_ratio = h / w if w > 0 else 1
            gender = "female" if aspect_ratio > 1.5 else "male" if aspect_ratio < 0.8 else "unknown"
        
        return {
            "gender": gender,
            "clothing": clothing,
        }
