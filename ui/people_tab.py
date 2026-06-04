from datetime import datetime
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, QFrame,
)


class PeopleLogItem(QWidget):
    """Custom widget for displaying a detected person."""
    
    def __init__(self, record: dict, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Face thumbnail
        face = record.get("face_crop")
        if face is not None and face.size > 0:
            h, w = face.shape[:2]
            if h > 0 and w > 0:
                ratio = 80 / max(h, w)
                thumb = cv2.resize(face, (int(w * ratio), int(h * ratio)))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                              rgb.shape[1] * 3, QImage.Format.Format_RGB888)
                thumb_label = QLabel()
                thumb_label.setPixmap(QPixmap.fromImage(qimg))
                thumb_label.setFixedSize(100, 100)
                layout.addWidget(thumb_label)
        else:
            placeholder = QLabel("No Face")
            placeholder.setFixedSize(100, 100)
            placeholder.setStyleSheet("border: 2px dashed #666; color: #999; background: #1a1a1a;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)
        
        # Person info
        info = QLabel(
            f"<b>Person #{record.get('person_id', '?')}</b><br>"
            f"Gender: {record.get('gender', '?').upper()}<br>"
            f"Clothing: {record.get('clothing', '?').upper()}<br>"
            f"Confidence: {record.get('confidence', 0.0):.2f}<br>"
            f"Frame: {record.get('frame', '?')}"
        )
        info.setWordWrap(True)
        layout.addWidget(info, 1)
        
        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(time_label)


class PeopleTab(QWidget):
    """Tab for people detection and analytics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: list[dict] = []
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Alert Panel (Hidden by default)
        self.alert_panel = QLabel("")
        self.alert_panel.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold; font-size: 20px; padding: 10px; border-radius: 5px;")
        self.alert_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_panel.setVisible(False)
        main_layout.addWidget(self.alert_panel)
        
        # Statistics panel
        stats_layout = QHBoxLayout()
        
        self.label_total = self._create_stat_box("Total", "0", "#FFD700")
        self.label_male = self._create_stat_box("Males", "0", "#4169E1")
        self.label_female = self._create_stat_box("Females", "0", "#FF1493")
        self.label_faces = self._create_stat_box("Faces", "0", "#32CD32")
        self.label_unknown = self._create_stat_box("Unknown", "0", "#808080")
        
        stats_layout.addWidget(self.label_total, 1)
        stats_layout.addWidget(self.label_male, 1)
        stats_layout.addWidget(self.label_female, 1)
        stats_layout.addWidget(self.label_faces, 1)
        stats_layout.addWidget(self.label_unknown, 1)
        
        stats_group = QGroupBox("People Statistics")
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        
        # Detection log and details in horizontal split
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Detection log
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        log_header = QLabel("Detection Log")
        log_header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        left_layout.addWidget(log_header)
        
        self.log_list = QListWidget()
        self.log_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.log_list)
        
        splitter.addWidget(left_panel)
        
        # Right: Details table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        details_header = QLabel("People Details")
        details_header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        right_layout.addWidget(details_header)
        
        self.details_table = QTableWidget(0, 6)
        self.details_table.setHorizontalHeaderLabels([
            "ID", "Gender", "Clothing", "Confidence", "Face Found", "Frame"
        ])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.details_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.details_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.details_table)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])  # Give much more space to the details table
        
        main_layout.addWidget(splitter, 1)
    
    def _create_stat_box(self, title: str, value: str, color: str) -> QGroupBox:
        """Create a statistic box."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        
        layout.addWidget(value_label)
        group.value_label = value_label
        
        return group
    
    def update_stats(self, stats: dict):
        """Update statistics display."""
        self.label_total.value_label.setText(str(stats.get("total_persons", 0)))
        self.label_male.value_label.setText(str(stats.get("males", 0)))
        self.label_female.value_label.setText(str(stats.get("females", 0)))
        self.label_faces.value_label.setText(str(stats.get("faces_captured", 0)))
        self.label_unknown.value_label.setText(str(stats.get("unknown", 0)))
        
        weapons = stats.get("weapons", [])
        if weapons:
            w_names = ", ".join(set([w["name"].upper() for w in weapons]))
            self.alert_panel.setText(f"🚨 ALERT: {w_names} DETECTED 🚨")
            self.alert_panel.setVisible(True)
        else:
            self.alert_panel.setVisible(False)
    
    def add_detection(self, record: dict):
        """Add a detection to the log and table."""
        self.results.append(record)
        self._add_log_item(record)
        self._add_table_row(record)
    
    def _add_log_item(self, record: dict):
        """Add item to detection log."""
        item = QListWidgetItem(self.log_list)
        widget = PeopleLogItem(record)
        item.setSizeHint(widget.sizeHint())
        self.log_list.addItem(item)
        self.log_list.setItemWidget(item, widget)
        self.log_list.scrollToBottom()
        
        # Keep only last 50 items
        if self.log_list.count() > 50:
            self.log_list.takeItem(0)
    
    def _add_table_row(self, record: dict):
        """Add row to details table."""
        row = self.details_table.rowCount()
        self.details_table.insertRow(row)
        self.details_table.setItem(row, 0, QTableWidgetItem(str(record.get("person_id", ""))))
        self.details_table.setItem(row, 1, QTableWidgetItem(str(record.get("gender", ""))))
        self.details_table.setItem(row, 2, QTableWidgetItem(str(record.get("clothing", ""))))
        self.details_table.setItem(row, 3, QTableWidgetItem(str(record.get("confidence", ""))))
        self.details_table.setItem(row, 4, QTableWidgetItem(
            "Yes" if record.get("face_found") else "No"
        ))
        self.details_table.setItem(row, 5, QTableWidgetItem(str(record.get("frame", ""))))
    
    def clear(self):
        """Clear all data."""
        self.results.clear()
        self.log_list.clear()
        self.details_table.setRowCount(0)
