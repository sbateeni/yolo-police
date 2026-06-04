import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import FRAME_SKIP, RESIZE_WIDTH, CAMERA_URL
from processor.pipeline import ALPRPipeline
from processor.people_pipeline import PeoplePipeline


class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    detection_event = pyqtSignal(dict)
    people_detection = pyqtSignal(dict)
    people_stats = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.is_live = False
        self._running = True
        self.pipeline = ALPRPipeline()
        self.people_pipeline = PeoplePipeline()
        self._last_result_count = 0
        self._last_people_count = 0
        self.frame_skip = FRAME_SKIP
        self.mode = "vehicle"

    def set_source(self, path: str, is_live: bool = False):
        self.source = path
        self.is_live = is_live

    def set_frame_skip(self, skip: int):
        self.frame_skip = max(1, int(skip))
    
    def set_mode(self, mode: str):
        self.mode = mode.lower()
        if self.mode == "people":
            self.people_pipeline = PeoplePipeline()
            self._last_people_count = 0
        else:
            self.pipeline = ALPRPipeline()
            self._last_result_count = 0

    def stop(self):
        self._running = False

    def run(self):
        source_url = self.source
        if isinstance(source_url, str) and ("earthcam.com" in source_url or "youtube.com" in source_url):
            try:
                import yt_dlp
                self.status_message.emit("Extracting stream URL via yt-dlp...")
                ydl_opts = {'format': 'best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source_url, download=False)
                    source_url = info.get('url', source_url)
            except ImportError:
                self.status_message.emit("yt-dlp not installed. Install via pip install yt-dlp")
            except Exception as e:
                self.status_message.emit(f"Extraction error: {str(e)}")

        cap = cv2.VideoCapture(source_url)
        if not cap.isOpened():
            self.status_message.emit(f"Failed to open: {source_url}")
            self.finished.emit()
            return

        if self.mode == "people":
            self.people_pipeline = PeoplePipeline()
            self._last_people_count = 0
        else:
            self.pipeline = ALPRPipeline()
            self._last_result_count = 0
        
        frame_idx = 0
        self._running = True
        mode_str = "People Detection" if self.mode == "people" else "Vehicle Detection"
        self.status_message.emit(f"Streaming [{mode_str}]: {self.source}")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if self.is_live:
                    self.status_message.emit("Reconnecting...")
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    continue
                break

            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            if self.mode == "people":
                display = self.people_pipeline.process_frame(frame, frame_idx)
                self.frame_ready.emit(display, frame_idx)
                
                for record in self.people_pipeline.results[self._last_people_count:]:
                    self.people_detection.emit(record)
                self._last_people_count = len(self.people_pipeline.results)
                
                stats = self.people_pipeline.get_stats()
                self.people_stats.emit(stats)
            else:
                display = self.pipeline.process_frame(frame, frame_idx)
                self.frame_ready.emit(display, frame_idx)

                for record in self.pipeline.results[self._last_result_count:]:
                    self.detection_event.emit(record)
                self._last_result_count = len(self.pipeline.results)
            
            frame_idx += 1

        cap.release()
        self.finished.emit()

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        return self.pipeline.process_frame(frame, frame_idx)

    def _border_color(self, v: dict) -> tuple:
        vtype = v.get("type", "")
        if vtype in ("police", "government", "ambulance", "fire"):
            return (255, 0, 0)
        if v.get("wanted", False):
            return (0, 0, 255)
        return (0, 255, 0)

    def _match_vehicle(self, plate_bbox: tuple, vehicles: list[dict]) -> dict | None:
        px1, py1, px2, py2 = plate_bbox
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                return v
        return None
