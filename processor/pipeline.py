import csv
import json
import logging

import cv2
import numpy as np

from config import (
    FRAME_SKIP, RESIZE_WIDTH, OUTPUT_CSV, OUTPUT_JSON, LOG_LEVEL,
    OCR_CONFIDENCE_THRESHOLD, MAX_PLATE_IMAGE_CACHE,
)
from processor.detector import VehicleDetector, PlateDetector
from utils.ocr import PlateReader

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ALPR")


class ALPRPipeline:
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.plate_reader = PlateReader()
        self.results: list[dict] = []
        self._crop_record_indexes: list[int] = []
        self._max_crop_cache = MAX_PLATE_IMAGE_CACHE

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        if FRAME_SKIP is None:
            frame_skip = 1
        else:
            frame_skip = FRAME_SKIP

        if frame_skip <= 0:
            frame_skip = 1

        if RESIZE_WIDTH:
            scale = RESIZE_WIDTH / orig_w
            new_w = RESIZE_WIDTH
            new_h = int(orig_h * scale)
            display = cv2.resize(frame, (new_w, new_h))
        else:
            display = frame.copy()

        scale_x = display.shape[1] / orig_w
        scale_y = display.shape[0] / orig_h

        vehicles = self.vehicle_detector.detect(frame)
        plates = self.plate_detector.detect(frame)

        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            label = f"{v['type']} ({v['color']}) {v['confidence']:.2f}"
            cv2.putText(display, label, (dx1, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for p in plates:
            x1, y1, x2, y2 = p["bbox"]
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            plate_text, ocr_confidence = self.plate_reader.read(p["crop"])
            plate_color = self.plate_detector.get_plate_color(p["crop"])
            is_clear = bool(plate_text and ocr_confidence >= OCR_CONFIDENCE_THRESHOLD)

            cv2.rectangle(display, (dx1, dy1), (dx2, dy2),
                          (0, 0, 255) if is_clear else (128, 128, 128),
                          2 if is_clear else 1)
            label = plate_text if plate_text else "UNCLEAR"
            cv2.putText(display, f"{label} ({plate_color})",
                        (dx1, dy2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255) if is_clear else (128, 128, 128), 2 if is_clear else 1)

            vehicle_match = self._match_vehicle(p["bbox"], vehicles)
            record = {
                "frame": frame_idx,
                "plate_text": plate_text,
                "plate_color": plate_color,
                "plate_confidence": p["confidence"],
                "ocr_confidence": round(ocr_confidence, 3),
                "is_clear": is_clear,
                "plate_bbox": p["bbox"],
                "vehicle_type": vehicle_match["type"] if vehicle_match else None,
                "vehicle_color": vehicle_match["color"] if vehicle_match else None,
                "vehicle_confidence": vehicle_match["confidence"] if vehicle_match else None,
                "vehicle_bbox": vehicle_match["bbox"] if vehicle_match else None,
                "plate_crop": p["crop"],
            }
            record_index = len(self.results)
            self.results.append(record)
            self._crop_record_indexes.append(record_index)
            if len(self._crop_record_indexes) > self._max_crop_cache:
                oldest_idx = self._crop_record_indexes.pop(0)
                if oldest_idx < len(self.results):
                    self.results[oldest_idx]["plate_crop"] = None

        return display

    def _match_vehicle(self, plate_bbox: tuple, vehicles: list[dict]) -> dict | None:
        px1, py1, px2, py2 = plate_bbox
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                return v
        return None

    def save_csv(self, path: str = OUTPUT_CSV):
        if not self.results:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [k for k in self.results[0].keys() if k != "plate_crop"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "plate_crop"} for row in self.results])
        log.info("Saved CSV: %s", path)

    def save_json(self, path: str = OUTPUT_JSON):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([
                {k: v for k, v in row.items() if k != "plate_crop"}
                for row in self.results
            ], f, ensure_ascii=False, indent=2)
        log.info("Saved JSON: %s", path)
