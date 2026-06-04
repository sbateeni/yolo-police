import argparse
import json
import csv
import sys
import logging
from pathlib import Path

import cv2
import numpy as np

from config import (
    DATA_DIR, FRAME_SKIP, RESIZE_WIDTH,
    OUTPUT_CSV, OUTPUT_JSON, LOG_LEVEL, CAMERA_URL,
    TAILSCALE_IP,
)
from processor.pipeline import ALPRPipeline

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ALPR")


def run_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", path)
        return

    pipeline = ALPRPipeline()
    frame_idx = 0
    log.info("Processing video: %s", path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        display = pipeline.process_frame(frame, frame_idx)
        cv2.imshow("ALPR - Video", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pipeline.save_csv()
    pipeline.save_json()


def run_camera(url: str = CAMERA_URL):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", url)
        return

    pipeline = ALPRPipeline()
    frame_idx = 0
    log.info("Live camera: %s", url)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(url)
            continue

        if frame_idx % FRAME_SKIP == 0:
            display = pipeline.process_frame(frame, frame_idx)
            cv2.imshow("ALPR - Live", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pipeline.save_csv()
    pipeline.save_json()


def run_gui():
    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("ALPR System")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(description="ALPR System - YOLOv8 + EasyOCR")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", type=str, help="Path to video file in data/")
    group.add_argument("--camera", action="store_true", help="Use live camera")
    group.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--camera-url", type=str, help="RTSP/HTTP camera URL")
    parser.add_argument("--list-data", action="store_true", help="List videos in data/")

    args = parser.parse_args()

    if args.list_data:
        files = list(DATA_DIR.glob("*.*"))
        if not files:
            print("No files in data/")
        else:
            for f in files:
                print(f.name)
        return

    if args.gui:
        run_gui()
        return

    if args.video:
        video_path = args.video if Path(args.video).is_absolute() else str(DATA_DIR / args.video)
        if not Path(video_path).exists():
            log.error("File not found: %s", video_path)
            return
        run_video(video_path)
    elif args.camera:
        url = args.camera_url or CAMERA_URL
        if TAILSCALE_IP:
            url = url.replace("192.168.", "100.")
        run_camera(url)


if __name__ == "__main__":
    main()
