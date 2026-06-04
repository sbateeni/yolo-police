from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QScrollArea, QWidget, QVBoxLayout, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import Qt
from config import CAMERA_URL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, FRAME_SKIP, OCR_CONFIDENCE_THRESHOLD

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ALPR Settings")
        self.setFixedSize(500, 600)
        
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QFormLayout(scroll_widget)
        scroll.setWidget(scroll_widget)

        # Camera settings
        self.camera_ip = QLineEdit(CAMERA_URL)
        layout.addRow("Camera URL:", self.camera_ip)

        # Detection settings
        self.conf_thresh = QDoubleSpinBox()
        self.conf_thresh.setRange(0.0, 1.0)
        self.conf_thresh.setValue(CONFIDENCE_THRESHOLD)
        self.conf_thresh.setSingleStep(0.05)
        self.conf_thresh.setDecimals(2)
        layout.addRow("Confidence Threshold:", self.conf_thresh)

        self.iou_thresh = QDoubleSpinBox()
        self.iou_thresh.setRange(0.0, 1.0)
        self.iou_thresh.setValue(IOU_THRESHOLD)
        self.iou_thresh.setSingleStep(0.05)
        self.iou_thresh.setDecimals(2)
        layout.addRow("IOU Threshold:", self.iou_thresh)

        # Performance settings
        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 30)
        self.frame_skip.setValue(FRAME_SKIP)
        layout.addRow("Frame Skip:", self.frame_skip)

        self.skip_enabled = QCheckBox("Enable Frame Skipping")
        self.skip_enabled.setChecked(True)
        layout.addRow(self.skip_enabled)

        # OCR settings
        self.ocr_conf_thresh = QDoubleSpinBox()
        self.ocr_conf_thresh.setRange(0.0, 1.0)
        self.ocr_conf_thresh.setValue(OCR_CONFIDENCE_THRESHOLD)
        self.ocr_conf_thresh.setSingleStep(0.05)
        self.ocr_conf_thresh.setDecimals(2)
        layout.addRow("OCR Confidence:", self.ocr_conf_thresh)

        self.ocr_languages = QLineEdit("en")
        self.ocr_languages.setPlaceholderText("e.g., en, ar, fr (separated by +)")
        layout.addRow("OCR Languages:", self.ocr_languages)

        # Output settings
        self.export_csv = QCheckBox("Export to CSV")
        self.export_csv.setChecked(True)
        layout.addRow(self.export_csv)

        self.export_json = QCheckBox("Export to JSON")
        self.export_json.setChecked(True)
        layout.addRow(self.export_json)

        self.enable_logging = QCheckBox("Enable Logging")
        self.enable_logging.setChecked(True)
        layout.addRow(self.enable_logging)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        main_layout.addWidget(scroll)
        main_layout.addWidget(buttons)

    def get_settings(self) -> dict:
        return {
            "camera_url": self.camera_ip.text(),
            "confidence": self.conf_thresh.value(),
            "iou_threshold": self.iou_thresh.value(),
            "frame_skip": self.frame_skip.value(),
            "frame_skip_enabled": self.skip_enabled.isChecked(),
            "ocr_confidence": self.ocr_conf_thresh.value(),
            "ocr_languages": self.ocr_languages.text(),
            "export_csv": self.export_csv.isChecked(),
            "export_json": self.export_json.isChecked(),
            "enable_logging": self.enable_logging.isChecked(),
        }
