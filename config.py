import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CAMERA_IP = os.getenv("CAMERA_IP", "192.168.1.100")
CAMERA_PORT = os.getenv("CAMERA_PORT", "554")
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "admin")
CAMERA_URL = os.getenv(
    "CAMERA_URL",
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:{CAMERA_PORT}/stream1",
)
EARTHCAM_URL = os.getenv(
    "EARTHCAM_URL",
    "https://www.youtube.com/watch?v=3nyPER2kzqk",
)

VEHICLE_MODEL_PATH = os.getenv(
    "VEHICLE_MODEL_PATH", str(MODELS_DIR / "yolov8n.pt")
)
PLATE_MODEL_PATH = os.getenv(
    "PLATE_MODEL_PATH", str(MODELS_DIR / "plate_detector.pt")
)
GENDER_MODEL_PATH = os.getenv(
    "GENDER_MODEL_PATH", str(MODELS_DIR / "yolov8n_gender.pt")
)
WEAPON_MODEL_PATH = os.getenv(
    "WEAPON_MODEL_PATH", str(MODELS_DIR / "yolov8_weapon.pt")
)
WANTED_PERSONS_DIR = os.getenv(
    "WANTED_PERSONS_DIR", str(DATA_DIR / "wanted_persons")
)
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en").split("+")
OCR_GPU = os.getenv("OCR_GPU", "1") == "1"
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.4"))
MAX_PLATE_IMAGE_CACHE = int(os.getenv("MAX_PLATE_IMAGE_CACHE", "50"))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "1280"))

VEHICLE_TYPES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic_light",
}

COLOR_NAME_THRESHOLD = 60
COLOR_PALETTE = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "beige": (220, 220, 200),
}

TAILSCALE_IP = os.getenv("TAILSCALE_IP", "")
STREAM_PORT = int(os.getenv("STREAM_PORT", "8080"))

OUTPUT_CSV = os.getenv("OUTPUT_CSV", str(DATA_DIR / "detections.csv"))
OUTPUT_JSON = os.getenv("OUTPUT_JSON", str(DATA_DIR / "detections.json"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Face Recognition Library: "deepface" (accurate, slower) or "face_recognition" (faster, real-time)
FACE_RECOGNITION_LIB = os.getenv("FACE_RECOGNITION_LIB", "deepface")

# DeepFace model options: "Facenet", "Facenet512", "VGG-Face", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "Facenet")
