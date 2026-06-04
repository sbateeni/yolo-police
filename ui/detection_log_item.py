import cv2
from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

class DetectionLogItem(QWidget):
    def __init__(self, record: dict, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        crop = record.get("plate_crop")
        if crop is not None and crop.size > 0:
            h, w = crop.shape[:2]
            if h > 0 and w > 0:
                ratio = 80 / max(h, w)
                thumb = cv2.resize(crop, (int(w * ratio), int(h * ratio)))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                              rgb.shape[1] * 3, QImage.Format.Format_RGB888)
                thumb_label = QLabel()
                thumb_label.setPixmap(QPixmap.fromImage(qimg))
                thumb_label.setFixedSize(100, 60)
                layout.addWidget(thumb_label)
        else:
            placeholder = QLabel("No image")
            placeholder.setFixedSize(100, 60)
            placeholder.setStyleSheet("border: 1px dashed #555; color: #888; margin-right: 6px;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)

        info = QLabel(
            f"<b>{record.get('plate_text', 'UNCLEAR')}</b><br>"
            f"OCR: {record.get('ocr_confidence', 0.0):.2f} | "
            f"Plate Color: {record.get('plate_color', '?')}<br>"
            f"Vehicle: {record.get('vehicle_type', '?')} "
            f"({record.get('vehicle_color', '?')})"
        )
        info.setWordWrap(True)
        layout.addWidget(info, 1)

        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setStyleSheet("color: gray;")
        layout.addWidget(time_label)
