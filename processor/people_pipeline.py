import csv
import json
import logging
import hashlib
import os
import glob

import cv2
import numpy as np

from config import (
    FRAME_SKIP, RESIZE_WIDTH, OUTPUT_CSV, OUTPUT_JSON, LOG_LEVEL,
    WEAPON_MODEL_PATH, WANTED_PERSONS_DIR, FACE_RECOGNITION_LIB, DEEPFACE_MODEL,
)
from processor.people_detector import PersonDetector

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("PeoplePipeline")


class PeoplePipeline:
    """Pipeline for detecting and tracking people with face capture."""

    def __init__(self):
        self.detector = PersonDetector()
        self.results: list[dict] = []
        self.face_hashes: dict[str, bool] = {}
        self.person_counter = 0
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0
        self.tracked_objects = {}
        self.max_distance = 100

        # Load Weapon Detection Model
        from ultralytics import YOLO
        if os.path.exists(WEAPON_MODEL_PATH):
            self.weapon_model = YOLO(WEAPON_MODEL_PATH)
        else:
            self.weapon_model = None
        self.current_weapons = []

        # Face Recognition setup
        self.wanted_dir = WANTED_PERSONS_DIR
        os.makedirs(self.wanted_dir, exist_ok=True)

        self.face_lib = FACE_RECOGNITION_LIB.lower()
        self.deepface_model = DEEPFACE_MODEL
        self.face_recognition_loaded = False
        self.deepface_loaded = False
        self.known_face_encodings = []
        self.known_face_names = []

        self._load_face_library()
        self._load_known_faces()

    def _load_face_library(self):
        """Load the selected face recognition library."""
        if self.face_lib == "face_recognition":
            try:
                import face_recognition
                self.face_recognition_loaded = True
                log.info("Loaded face_recognition library")
            except ImportError:
                log.warning("face_recognition not installed, falling back to deepface")
                self.face_lib = "deepface"

        if self.face_lib == "deepface":
            try:
                import deepface
                self.deepface_loaded = True
                log.info("Loaded deepface library")
            except ImportError:
                log.warning("deepface not installed")

    def _load_known_faces(self):
        """Pre-load known faces for face_recognition library."""
        if self.face_lib != "face_recognition" or not self.face_recognition_loaded:
            return

        import face_recognition
        image_files = glob.glob(os.path.join(self.wanted_dir, "*.*"))
        for img_path in image_files:
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.basename(img_path).split('.')[0]
                    self.known_face_names.append(name)
                    log.info(f"Loaded known face: {name}")
            except Exception as e:
                log.warning(f"Failed to load {img_path}: {e}")

    def _check_wanted_deepface(self, face_crop: np.ndarray) -> tuple[bool, str]:
        """Check if face matches wanted person using DeepFace."""
        if not self.deepface_loaded:
            return False, ""

        try:
            from deepface import DeepFace
            image_files = glob.glob(os.path.join(self.wanted_dir, "*.*"))
            if not image_files:
                return False, ""

            dfs = DeepFace.find(
                img_path=face_crop,
                db_path=self.wanted_dir,
                enforce_detection=False,
                silent=True,
                model_name=self.deepface_model
            )
            if len(dfs) > 0 and not dfs[0].empty:
                wanted_name = os.path.basename(dfs[0].iloc[0]['identity']).split('.')[0]
                return True, wanted_name
        except Exception as e:
            log.debug(f"DeepFace error: {e}")
        return False, ""

    def _check_wanted_face_recognition(self, face_crop: np.ndarray) -> tuple[bool, str]:
        """Check if face matches wanted person using face_recognition."""
        if not self.face_recognition_loaded or not self.known_face_encodings:
            return False, ""

        try:
            import face_recognition
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)
            if not face_encodings:
                return False, ""

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0], tolerance=0.5)
            if True in matches:
                idx = matches.index(True)
                return True, self.known_face_names[idx]
        except Exception as e:
            log.debug(f"face_recognition error: {e}")
        return False, ""

    def _check_wanted(self, face_crop: np.ndarray) -> tuple[bool, str]:
        """Check if face matches any wanted person."""
        if face_crop is None or face_crop.size == 0:
            return False, ""

        if self.face_lib == "face_recognition":
            return self._check_wanted_face_recognition(face_crop)
        else:
            return self._check_wanted_deepface(face_crop)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process frame and detect persons."""
        orig_h, orig_w = frame.shape[:2]

        if RESIZE_WIDTH:
            scale = RESIZE_WIDTH / orig_w
            new_w = RESIZE_WIDTH
            new_h = int(orig_h * scale)
            display = cv2.resize(frame, (new_w, new_h))
        else:
            display = frame.copy()

        scale_x = display.shape[1] / orig_w
        scale_y = display.shape[0] / orig_h

        self.current_weapons = []
        if self.weapon_model is not None:
            w_results = self.weapon_model(frame, conf=0.45, verbose=False)[0]
            for box in w_results.boxes:
                wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                w_conf = float(box.conf[0])
                w_cls = int(box.cls[0])
                w_name = w_results.names.get(w_cls, "weapon")

                dwx1, dwy1 = int(wx1 * scale_x), int(wy1 * scale_y)
                dwx2, dwy2 = int(wx2 * scale_x), int(wy2 * scale_y)

                self.current_weapons.append({"name": w_name, "confidence": w_conf})

                cv2.rectangle(display, (dwx1, dwy1), (dwx2, dwy2), (0, 0, 255), 4)
                cv2.putText(display, f"ALERT: {w_name.upper()}", (dwx1, dwy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        persons = self.detector.detect_persons(frame)

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            # Centroid tracking
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            matched_id = None
            min_dist = float('inf')
            for pid, track in self.tracked_objects.items():
                tcx, tcy = track["centroid"]
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched_id = pid

            if matched_id is None:
                self.person_counter += 1
                matched_id = self.person_counter
                is_new = True
            else:
                is_new = False

            self.tracked_objects[matched_id] = {
                "centroid": (cx, cy),
                "last_seen": frame_idx
            }

            # Draw person box (yellow default, red if wanted)
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)

            # Extract faces
            faces = self.detector.extract_faces(p["crop"])

            # Capture best face (first clear face without duplicate)
            best_face = None
            for face in faces:
                face_crop = face["crop"]
                if face_crop.size > 0:
                    face_hash = self._hash_face(face_crop)
                    if face_hash not in self.face_hashes:
                        self.face_hashes[face_hash] = True
                        best_face = face_crop
                        break

            # Estimate attributes
            attrs = self.detector.estimate_gender_clothing(p["crop"], best_face)
            gender = attrs["gender"]
            clothing = attrs["clothing"]

            # Update counters ONLY IF NEW
            if is_new:
                if gender == "male":
                    self.male_count += 1
                elif gender == "female":
                    self.female_count += 1
                else:
                    self.unknown_count += 1

            # Create record only if new person or new face
            if is_new or best_face is not None:
                wanted_status = False
                wanted_name = ""

                # Check against wanted database
                if best_face is not None:
                    wanted_status, wanted_name = self._check_wanted(best_face)
                    if wanted_status:
                        self.current_weapons.append({"name": f"WANTED PERSON: {wanted_name}", "confidence": 1.0})

                record = {
                    "frame": frame_idx,
                    "person_id": matched_id,
                    "gender": gender,
                    "clothing": clothing,
                    "confidence": p["confidence"],
                    "bbox": p["bbox"],
                    "face_crop": best_face,
                    "face_found": best_face is not None,
                    "wanted": wanted_status,
                    "wanted_name": wanted_name
                }
                self.results.append(record)

            # Draw label on display
            is_wanted = self.results and self.results[-1].get("wanted", False)
            color = (0, 0, 255) if is_wanted else (0, 255, 255)
            if is_wanted:
                label = f"WANTED: {self.results[-1].get('wanted_name', '')}"
            else:
                label = f"{gender} - {clothing} (#{matched_id})"
            cv2.putText(display, label, (dx1, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Cleanup old tracks
        old_ids = []
        for pid, track in self.tracked_objects.items():
            if frame_idx - track["last_seen"] > 30:
                old_ids.append(pid)
        for pid in old_ids:
            del self.tracked_objects[pid]

        return display

    def _hash_face(self, face_crop: np.ndarray) -> str:
        """Generate hash for face to detect duplicates."""
        small_face = cv2.resize(face_crop, (10, 10))
        face_bytes = small_face.tobytes()
        return hashlib.md5(face_bytes).hexdigest()

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "total_persons": self.person_counter,
            "males": self.male_count,
            "females": self.female_count,
            "unknown": self.unknown_count,
            "faces_captured": len(self.face_hashes),
            "weapons": self.current_weapons,
        }

    def save_csv(self, path: str = None):
        """Save results to CSV (without face_crop)."""
        if not self.results:
            return

        if path is None:
            path = str(__file__).replace("people_pipeline.py", "") + "/../data/people_detections.csv"

        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [k for k in self.results[0].keys() if k != "face_crop"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "face_crop"} for row in self.results])

        log.info("Saved people CSV: %s", path)

    def save_json(self, path: str = None):
        """Save results to JSON (without face_crop)."""
        if path is None:
            path = str(__file__).replace("people_pipeline.py", "") + "/../data/people_detections.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump([
                {k: v for k, v in row.items() if k != "face_crop"}
                for row in self.results
            ], f, ensure_ascii=False, indent=2)

        log.info("Saved people JSON: %s", path)