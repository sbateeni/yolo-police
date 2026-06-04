from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox
from config import CAMERA_URL

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ALPR Settings")
        self.setFixedSize(400, 300)
        layout = QFormLayout(self)

        self.camera_ip = QLineEdit(CAMERA_URL)
        layout.addRow("Camera URL:", self.camera_ip)

        self.conf_thresh = QLineEdit("0.5")
        layout.addRow("Confidence:", self.conf_thresh)

        self.frame_skip = QCheckBox("Skip frames")
        self.frame_skip.setChecked(True)
        layout.addRow(self.frame_skip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self) -> dict:
        return {
            "camera_url": self.camera_ip.text(),
            "confidence": float(self.conf_thresh.text()),
            "frame_skip": self.frame_skip.isChecked(),
        }
