import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QSplitter, QFileDialog, QMessageBox, QStatusBar,
    QFrame, QComboBox, QLineEdit, QGroupBox, QFormLayout,
    QDialog, QDialogButtonBox, QCheckBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from config import DATA_DIR, CAMERA_URL, OUTPUT_CSV, OUTPUT_JSON, FRAME_SKIP, EARTHCAM_URL
from ui.workers import VideoWorker
from ui.people_tab import PeopleTab


from ui.settings_dialog import SettingsDialog
from ui.detection_log_item import DetectionLogItem

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALPR System - Automatic License Plate Recognition")
        self.setMinimumSize(1280, 800)

        self._setup_ui()
        self._setup_menu()
        self._setup_style()

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.detection_event.connect(self.add_detection)
        self.worker.people_detection.connect(self.add_people_detection)
        self.worker.people_stats.connect(self.update_people_stats)
        self.worker.status_message.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_worker_finished)

        self.frame_skip_enabled = True
        self.frame_skip_count = FRAME_SKIP
        self.worker.set_frame_skip(self.frame_skip_count)
        self.current_mode = "vehicle"

        self.results: list[dict] = []
        self.current_source = None
        self.is_live = False
        self.camera_url = CAMERA_URL

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("No video source")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; border: 2px solid #333; border-radius: 8px;"
        )
        left_layout.addWidget(self.video_label, 1)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(60)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Vehicle (ALPR)", "People Detection"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.mode_combo.setObjectName("modeCombo")  # Set object name for styling
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        left_layout.addLayout(mode_layout)

        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_earthcam = QPushButton("EarthCam")
        self.btn_earthcam.clicked.connect(self.start_earthcam)
        self.btn_file = QPushButton("Open Video")
        self.btn_file.clicked.connect(self.open_video)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_lock = QPushButton("Lock")
        self.btn_lock.setCheckable(True)
        self.btn_lock.toggled.connect(self.toggle_lock)

        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_earthcam)
        controls.addWidget(self.btn_file)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_export)
        controls.addWidget(self.btn_settings)
        controls.addWidget(self.btn_lock)
        left_layout.addLayout(controls)

        splitter.addWidget(left_panel)

        # Tabbed interface for different modes
        self.tabs = QTabWidget()
        
        # Vehicle tab
        vehicle_widget = QWidget()
        vehicle_layout = QVBoxLayout(vehicle_widget)
        vehicle_layout.setContentsMargins(0, 0, 0, 0)
        
        search_bar = QHBoxLayout()
        search_label = QLabel("Search plate:")
        search_label.setFixedWidth(90)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter plate text or vehicle type")
        self.search_input.textChanged.connect(self.filter_results)
        search_bar.addWidget(search_label)
        search_bar.addWidget(self.search_input)
        vehicle_layout.addLayout(search_bar)

        log_header = QLabel("Detection Log")
        log_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px;"
        )
        vehicle_layout.addWidget(log_header)

        self.log_list = QListWidget()
        self.log_list.setAlternatingRowColors(True)
        vehicle_layout.addWidget(self.log_list, 1)

        details_header = QLabel("Vehicle Details")
        details_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px;"
        )
        vehicle_layout.addWidget(details_header)

        self.details_table = QTableWidget(0, 6)
        self.details_table.setHorizontalHeaderLabels([
            "Plate", "OCR", "Vehicle", "Vehicle Color", "Plate Color", "Frame"
        ])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.details_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.details_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        vehicle_layout.addWidget(self.details_table, 1)
        
        self.tabs.addTab(vehicle_widget, "Vehicles (ALPR)")
        
        # People tab
        self.people_tab = PeopleTab()
        self.tabs.addTab(self.people_tab, "People Detection")
        
        splitter.addWidget(self.tabs)
        splitter.setSizes([860, 420])
        main_layout.addWidget(splitter, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video...", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        export_action = QAction("Export CSV", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        stream_menu = menubar.addMenu("Stream")
        
        start_action = QAction("Start Camera", self)
        start_action.triggered.connect(self.start_camera)
        stream_menu.addAction(start_action)
        
        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_stream)
        stream_menu.addAction(stop_action)

    def _setup_style(self):
        self.setStyleSheet("""
            /* Modern Dark Theme for PyQt */
            QMainWindow {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #1f1f1f;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                border: 1px solid #0A84FF;
            }
            QPushButton:pressed {
                background-color: #0A84FF;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #0A84FF;
                color: #ffffff;
                border: 1px solid #005bb5;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #222222;
            }
            QLineEdit, QComboBox {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 13px;
            }
            #modeCombo {
                color: #0A84FF;
                font-weight: bold;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #0A84FF;
            }
            QListWidget {
                background-color: #181818;
                color: #e0e0e0;
                border: 1px solid #2b2b2b;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #222;
            }
            QListWidget::item:alternate {
                background-color: #1c1c1c;
            }
            QListWidget::item:selected {
                background-color: #0A84FF;
                color: #ffffff;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #181818;
                color: #e0e0e0;
                border: 1px solid #2b2b2b;
                border-radius: 6px;
                gridline-color: #2b2b2b;
            }
            QHeaderView::section {
                background-color: #232323;
                color: #ffffff;
                padding: 6px;
                border: none;
                border-right: 1px solid #2b2b2b;
                border-bottom: 1px solid #2b2b2b;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #0A84FF;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                background: #181818;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #1f1f1f;
                color: #888;
                border: 1px solid #333;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #0A84FF;
                color: white;
                border-color: #0A84FF;
            }
            QTabBar::tab:hover:!selected {
                background: #2a2a2a;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                color: #0A84FF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QStatusBar {
                background-color: #121212;
                color: #888888;
                border-top: 1px solid #222;
            }
            QMenuBar {
                background-color: #121212;
                color: #e0e0e0;
                border-bottom: 1px solid #222;
            }
            QMenuBar::item:selected {
                background-color: #1f1f1f;
            }
            QMenu {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #333;
            }
            QMenu::item:selected {
                background-color: #0A84FF;
            }
            QSplitter::handle {
                background-color: #333;
                width: 2px;
            }
        """)

    def start_camera(self):
        self.stop_stream()
        self.current_source = self.camera_url
        self.is_live = True
        self.worker.set_frame_skip(self.frame_skip_count)
        self.worker.set_source(self.current_source, is_live=True)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage(f"Camera started: {self.current_source}")

    def start_earthcam(self):
        self.stop_stream()
        self.current_source = EARTHCAM_URL
        self.is_live = True
        self.worker.set_frame_skip(self.frame_skip_count)
        self.worker.set_source(self.current_source, is_live=True)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("EarthCam started")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", str(DATA_DIR),
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*.*)"
        )
        if not path:
            return
        self.stop_stream()
        self.current_source = path
        self.is_live = False
        self.worker.set_frame_skip(self.frame_skip_count)
        self.worker.set_source(path)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage(f"Playing: {Path(path).name}")

    def stop_stream(self):
        self.worker.stop()
        self.worker.wait(2000)
        self.btn_start.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stopped")

    def update_frame(self, frame: np.ndarray, frame_idx: int):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def add_detection(self, record: dict):
        self.results.append(record)
        self._add_log_item(record)
        if self.search_input.text().strip() == "":
            self._add_table_row(record)
        else:
            self.filter_results(self.search_input.text())

    def _add_log_item(self, record: dict):
        item = QListWidgetItem(self.log_list)
        widget = DetectionLogItem(record)
        item.setSizeHint(widget.sizeHint())
        self.log_list.addItem(item)
        self.log_list.setItemWidget(item, widget)
        self.log_list.scrollToBottom()
        if self.log_list.count() > 100:
            self.log_list.takeItem(0)

    def _add_table_row(self, record: dict, visible: bool = True):
        row = self.details_table.rowCount()
        self.details_table.insertRow(row)
        self.details_table.setItem(row, 0, QTableWidgetItem(record.get("plate_text", "")))
        self.details_table.setItem(row, 1, QTableWidgetItem(str(record.get("ocr_confidence", ""))))
        self.details_table.setItem(row, 2, QTableWidgetItem(str(record.get("vehicle_type", ""))))
        self.details_table.setItem(row, 3, QTableWidgetItem(str(record.get("vehicle_color", ""))))
        self.details_table.setItem(row, 4, QTableWidgetItem(str(record.get("plate_color", ""))))
        self.details_table.setItem(row, 5, QTableWidgetItem(str(record.get("frame", ""))))
        self.details_table.setRowHidden(row, not visible)

    def _clear_table(self):
        self.details_table.setRowCount(0)

    def filter_results(self, text: str):
        query = text.strip().lower()
        self._clear_table()
        for record in self.results:
            if not query:
                self._add_table_row(record, visible=True)
                continue
            searchable = (
                str(record.get("plate_text", "")).lower() + " " +
                str(record.get("vehicle_type", "")).lower() + " " +
                str(record.get("vehicle_color", "")).lower() + " " +
                str(record.get("plate_color", "")).lower()
            )
            if query in searchable:
                self._add_table_row(record, visible=True)

    def export_results(self):
        if not self.results:
            QMessageBox.information(self, "Export", "No results to export.")
            return

        csv_path = OUTPUT_CSV
        json_path = OUTPUT_JSON

        clean = []
        for r in self.results:
            clean.append({k: v for k, v in r.items() if k != "plate_crop"})

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=clean[0].keys())
            writer.writeheader()
            writer.writerows(clean)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)

        QMessageBox.information(
            self, "Export Done",
            f"CSV: {csv_path}\nJSON: {json_path}\nRecords: {len(clean)}"
        )

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.camera_ip.setText(self.camera_url)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self.apply_settings(settings)
            QMessageBox.information(self, "Settings", "Settings saved.")

    def apply_settings(self, settings: dict):
        self.camera_url = settings.get("camera_url", self.camera_url)
        self.frame_skip_enabled = settings.get("frame_skip_enabled", True)
        self.frame_skip_count = settings.get("frame_skip", FRAME_SKIP) if self.frame_skip_enabled else 1
        self.worker.set_frame_skip(self.frame_skip_count)
        
        # Pass settings to worker for detection parameters
        detection_settings = {
            "confidence": settings.get("confidence", 0.5),
            "iou_threshold": settings.get("iou_threshold", 0.45),
            "ocr_confidence": settings.get("ocr_confidence", 0.4),
            "ocr_languages": settings.get("ocr_languages", "en"),
        }
        self.worker.set_detection_settings(detection_settings)
        
        message = (
            f"Settings saved" if settings else "Settings saved"
        )
        self.status_bar.showMessage(message)

    def toggle_lock(self, checked):
        if checked:
            self.btn_lock.setText("Unlock")
            self.btn_start.setEnabled(False)
            self.btn_file.setEnabled(False)
            self.btn_settings.setEnabled(False)
            self.btn_export.setEnabled(False)
            self.status_bar.showMessage("SYSTEM LOCKED")
        else:
            self.btn_lock.setText("Lock")
            self.btn_start.setEnabled(True)
            self.btn_file.setEnabled(True)
            self.btn_settings.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.status_bar.showMessage("Unlocked")

    def on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stream ended")

    def on_mode_changed(self, mode_text: str):
        """Switch between vehicle and people detection modes."""
        if "People" in mode_text:
            self.current_mode = "people"
            self.tabs.setCurrentIndex(1)
        else:
            self.current_mode = "vehicle"
            self.tabs.setCurrentIndex(0)
        
        self.worker.set_mode(self.current_mode)
        self.status_bar.showMessage(f"Mode: {mode_text}")

    def add_people_detection(self, record: dict):
        """Add a people detection to the people tab."""
        self.people_tab.add_detection(record)

    def update_people_stats(self, stats: dict):
        """Update people statistics in the people tab."""
        self.people_tab.update_stats(stats)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()
