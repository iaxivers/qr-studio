"""
QR Studio
Framework: PySide6
Deps: qrcode, Pillow, opencv-python, numpy
"""

import sys, os, re
from datetime import datetime
from urllib.parse import quote

from PySide6.QtCore import Qt, QSize, QTimer, QThread, QObject, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPixmap, QImage, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QFrame, QStatusBar, QListWidget, QListWidgetItem,
    QTextEdit, QLineEdit, QComboBox, QCheckBox, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea
)

# third-party
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image, ImageQt
import cv2
import numpy as np


# ---------------- Palette ----------------
PAL = {
    "bg": "#1e1f24",
    "card": "#2a2d32",
    "border": "#3a3d44",
    "text": "#d1d5db",
    "muted": "#9aa3b2",
    "accent": "#8b5cf6"
}


# ---------------- Utils ----------------
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(ImageQt.ImageQt(img))


def cv_to_qimage(frame) -> QImage:
    """Convert OpenCV BGR/BGRA frame to QImage."""
    if frame is None:
        return QImage()
    if len(frame.shape) == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = frame.shape
    if ch == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    if ch == 4:
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        return QImage(rgba.data, w, h, ch * w, QImage.Format_RGBA8888).copy()
    # Fallback to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def ensure_http(url: str) -> str:
    if not url:
        return ""
    if re.match(r"^https?://", url, re.I):
        return url
    return "http://" + url


# Escapers for payload formats
def _wifi_escape(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\\", r"\\")
    return re.sub(r'([;,:"])', r'\\\1', s)

def _ics_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("\\", r"\\")
         .replace("\n", r"\n")
         .replace(",", r"\,")
         .replace(";", r"\;")
    )

def _vc_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("\\", r"\\")
         .replace("\n", r"\n")
         .replace(";", r"\;")
         .replace(",", r"\,")
    )


# ---------------- Small UI helpers ----------------
class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFrameShape(QFrame.NoFrame)


class HeaderLabel(QLabel):
    """Uniform header bar used across all cards."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("SectionTitle")
        self.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.setFixedHeight(28)  # consistent across app
        self.setMargin(6)



# ---------------- Camera worker ----------------
class CameraOpener(QObject):
    opened = Signal(object, str)  # (cap or None, error)
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
    def run(self):
        try:
            # Try DirectShow (faster on many Windows setups)
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                # Fallback: default backend
                cap = cv2.VideoCapture(self.index)
            if not cap or not cap.isOpened():
                self.opened.emit(None, "Failed to open camera")
                return
            # lower resolution for faster init
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.opened.emit(cap, "")
        except Exception as e:
            self.opened.emit(None, str(e))


# ---------------- Generate Tab ----------------
class GenerateTab(QWidget):
    def __init__(self, status_cb):
        super().__init__()
        self.status_cb = status_cb
        self.logo_img: Image.Image | None = None
        self.current_img: Image.Image | None = None

        self._build_ui()

        # live preview every 250 ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(250)

    # ---- UI
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        # Left: type list
        type_card = Card()
        tl = QVBoxLayout(type_card)
        tl.setContentsMargins(8, 8, 8, 8)
        tl.setSpacing(8)
        tl.addWidget(HeaderLabel("QR Type"))

        self.type_list = QListWidget()
        for t in ["Text / URL", "Wi-Fi", "SMS", "Email", "Event", "vCard"]:
            self.type_list.addItem(QListWidgetItem(t))
        self.type_list.setCurrentRow(0)
        self.type_list.currentRowChanged.connect(self._rebuild_fields)

        tl.addWidget(self.type_list)

        # Middle: content fields with header + scroll area
        self.content_card = Card()
        cc = QVBoxLayout(self.content_card)
        cc.setContentsMargins(8, 8, 8, 8)
        cc.setSpacing(8)
        cc.addWidget(HeaderLabel("Content"))

        self.fields_area = QScrollArea()
        self.fields_area.setWidgetResizable(True)
        self.fields_host = QWidget()
        self.fields_layout = QVBoxLayout(self.fields_host)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        self.fields_layout.setSpacing(6)
        self.fields_area.setWidget(self.fields_host)
        cc.addWidget(self.fields_area)

        # Right: preview
        prev_card = Card()
        pl = QVBoxLayout(prev_card)
        pl.setContentsMargins(8, 8, 8, 8)
        pl.setSpacing(8)
        pl.addWidget(HeaderLabel("Live Preview"))

        self.preview = QLabel("Preview area")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(280, 280)
        self.preview.setStyleSheet("border:1px dashed #3a3d44;")
        pl.addWidget(self.preview, 1)

        row = QHBoxLayout()
        self.btn_gen = QPushButton("Generate"); self.btn_gen.setObjectName("Primary")
        self.btn_save = QPushButton("Save Image…")
        self.btn_copy = QPushButton("Copy Payload")
        self.btn_copy_png = QPushButton("Copy PNG")
        self.btn_logo = QPushButton("Add Logo…")
        for b in (self.btn_gen, self.btn_save, self.btn_copy, self.btn_copy_png, self.btn_logo):
            row.addWidget(b)
        row.addStretch(1)
        self.btn_gen.clicked.connect(self.update_preview)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_copy.clicked.connect(self.copy_payload)
        self.btn_copy_png.clicked.connect(self.copy_png)
        self.btn_logo.clicked.connect(self.import_logo)
        pl.addLayout(row)

        root.addWidget(type_card, 1)
        root.addWidget(self.content_card, 2)
        root.addWidget(prev_card, 3)

        # make all inputs
        self._create_inputs()
        self._rebuild_fields(0)

        # start with disabled buttons until we have content
        self.btn_save.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.btn_copy_png.setEnabled(False)

    def _create_inputs(self):
        # text/url
        self.in_text = QTextEdit()
        self.in_text.setPlaceholderText("Enter text or URL…")
        self.in_text.setMinimumHeight(120)

        # wifi
        self.wifi_ssid = QLineEdit(); self.wifi_ssid.setPlaceholderText("SSID")
        self.wifi_pass = QLineEdit(); self.wifi_pass.setPlaceholderText("Password"); self.wifi_pass.setEchoMode(QLineEdit.Password)
        self.wifi_auth = QComboBox(); self.wifi_auth.addItems(["WPA/WPA2", "WEP", "nopass"])
        self.wifi_hidden = QCheckBox("Hidden network")

        # sms
        self.sms_phone = QLineEdit(); self.sms_phone.setPlaceholderText("Phone number")
        self.sms_msg = QTextEdit(); self.sms_msg.setPlaceholderText("Message"); self.sms_msg.setMinimumHeight(100)

        # email
        self.email_to = QLineEdit(); self.email_to.setPlaceholderText("to@example.com")
        self.email_subj = QLineEdit(); self.email_subj.setPlaceholderText("Subject")
        self.email_body = QTextEdit(); self.email_body.setPlaceholderText("Body"); self.email_body.setMinimumHeight(120)

        # event
        self.ev_title = QLineEdit(); self.ev_title.setPlaceholderText("Title")
        self.ev_loc = QLineEdit(); self.ev_loc.setPlaceholderText("Location")
        self.ev_start = QLineEdit(); self.ev_start.setPlaceholderText("Start (YYYYMMDDTHHMMSS or YYYYMMDD)")
        self.ev_end = QLineEdit(); self.ev_end.setPlaceholderText("End (YYYYMMDDTHHMMSS or YYYYMMDD)")
        self.ev_desc = QTextEdit(); self.ev_desc.setPlaceholderText("Description"); self.ev_desc.setMinimumHeight(100)

        # vcard
        self.vc_name = QLineEdit(); self.vc_name.setPlaceholderText("Full name")
        self.vc_org = QLineEdit(); self.vc_org.setPlaceholderText("Organization")
        self.vc_title = QLineEdit(); self.vc_title.setPlaceholderText("Title")
        self.vc_phone = QLineEdit(); self.vc_phone.setPlaceholderText("Phone")
        self.vc_email = QLineEdit(); self.vc_email.setPlaceholderText("Email")
        self.vc_url = QLineEdit(); self.vc_url.setPlaceholderText("Website")

    def _rebuild_fields(self, idx: int):
        # clear fields container (not the header)
        while self.fields_layout.count():
            item = self.fields_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        if idx == 0:  # Text/URL
            self.fields_layout.addWidget(self.in_text)
        elif idx == 1:  # Wi-Fi
            for w in [self.wifi_ssid, self.wifi_pass, self.wifi_auth, self.wifi_hidden]:
                self.fields_layout.addWidget(w)
        elif idx == 2:  # SMS
            for w in [self.sms_phone, self.sms_msg]:
                self.fields_layout.addWidget(w)
        elif idx == 3:  # Email
            for w in [self.email_to, self.email_subj, self.email_body]:
                self.fields_layout.addWidget(w)
        elif idx == 4:  # Event
            for w in [self.ev_title, self.ev_loc, self.ev_start, self.ev_end, self.ev_desc]:
                self.fields_layout.addWidget(w)
        elif idx == 5:  # vCard
            for w in [self.vc_name, self.vc_org, self.vc_title, self.vc_phone, self.vc_email, self.vc_url]:
                self.fields_layout.addWidget(w)

        self.fields_layout.addStretch(1)

    # ---- payload + render
    def _ensure_z(self, ts: str) -> str:
        if not ts:
            return ""
        if ts.endswith("Z"):
            return ts
        # allow YYYYMMDD too
        if re.fullmatch(r"\d{8}$", ts):
            return ts  # all-day date
        return ts + "Z"

    def build_payload(self) -> str:
        idx = self.type_list.currentRow()
        if idx == 0:
            return self.in_text.toPlainText().strip()

        if idx == 1:
            ssid = _wifi_escape(self.wifi_ssid.text().strip())
            pw_raw = self.wifi_pass.text()
            auth = self.wifi_auth.currentText()
            t = "WPA" if "WPA" in auth else ("WEP" if auth == "WEP" else "nopass")
            parts = [f"WIFI:T:{t}", f"S:{ssid}"]
            if t != "nopass" and pw_raw:
                parts.append(f"P:{_wifi_escape(pw_raw)}")
            if self.wifi_hidden.isChecked():
                parts.append("H:true")
            return ";".join(parts) + ";"

        if idx == 2:
            return f"SMSTO:{self.sms_phone.text().strip()}:{self.sms_msg.toPlainText().strip()}"

        if idx == 3:
            to = self.email_to.text().strip()
            subj = quote(self.email_subj.text().strip(), safe='')
            body = quote(self.email_body.toPlainText(), safe='')
            return f"mailto:{to}?subject={subj}&body={body}"

        if idx == 4:
            start_raw = self.ev_start.text().strip()
            end_raw = self.ev_end.text().strip()
            lines = [
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "BEGIN:VEVENT",
                f"SUMMARY:{_ics_escape(self.ev_title.text().strip())}",
            ]
            if start_raw:
                lines.append(f"DTSTART:{self._ensure_z(start_raw)}")
            if end_raw:
                lines.append(f"DTEND:{self._ensure_z(end_raw)}")
            if self.ev_loc.text().strip():
                lines.append(f"LOCATION:{_ics_escape(self.ev_loc.text().strip())}")
            desc = self.ev_desc.toPlainText().strip()
            if desc:
                lines.append("DESCRIPTION:" + _ics_escape(desc))
            lines += ["END:VEVENT", "END:VCALENDAR"]
            return "\n".join(lines)

        if idx == 5:
            fn = self.vc_name.text().strip()
            name_parts = fn.split()
            last = name_parts[-1] if name_parts else ""
            first = " ".join(name_parts[:-1]) if len(name_parts) > 1 else ""
            n_field = f"{last};{first};;;"
            lines = [
                "BEGIN:VCARD",
                "VERSION:3.0",
                f"FN:{_vc_escape(fn)}",
                f"N:{_vc_escape(n_field)}"
            ]
            if self.vc_org.text().strip():   lines.append(f"ORG:{_vc_escape(self.vc_org.text().strip())}")
            if self.vc_title.text().strip(): lines.append(f"TITLE:{_vc_escape(self.vc_title.text().strip())}")
            if self.vc_phone.text().strip(): lines.append(f"TEL;TYPE=CELL:{_vc_escape(self.vc_phone.text().strip())}")
            if self.vc_email.text().strip(): lines.append(f"EMAIL:{_vc_escape(self.vc_email.text().strip())}")
            if self.vc_url.text().strip():   lines.append(f"URL:{_vc_escape(ensure_http(self.vc_url.text().strip()))}")
            lines.append("END:VCARD")
            return "\n".join(lines)

        return ""

    def update_preview(self):
        payload = self.build_payload()
        if not payload:
            self.preview.setText("Preview area")
            self.current_img = None
            self.btn_save.setEnabled(False)
            self.btn_copy.setEnabled(False)
            self.btn_copy_png.setEnabled(False)
            return

        qr = qrcode.QRCode(
            version=None,
            error_correction=ERROR_CORRECT_H,  # safe with logo overlay
            box_size=8,
            border=2
        )
        try:
            qr.add_data(payload)
            qr.make(fit=True)
        except Exception as e:
            QMessageBox.warning(self, "QR Error", str(e))
            self.status_cb(f"QR error: {e}")
            return

        img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

        if self.logo_img:
            logo = self.logo_img.copy().convert("RGBA")
            w, h = img.size
            target = int(min(w, h) * 0.20)
            logo.thumbnail((target, target))
            lw, lh = logo.size

            # white backing behind logo for contrast
            bg = Image.new("RGBA", (lw + 12, lh + 12), (255, 255, 255, 255))
            bg.paste(logo, (6, 6), logo)
            img.alpha_composite(bg, dest=((w - bg.width) // 2, (h - bg.height) // 2))

        self.current_img = img
        self.preview.setPixmap(
            pil_to_qpixmap(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # enable actions when we have something meaningful
        self.btn_save.setEnabled(True)
        self.btn_copy.setEnabled(bool(payload))
        self.btn_copy_png.setEnabled(True)

    def save_image(self):
        if not self.current_img:
            return

        filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.ico);;All Files (*)"
        sugg = f"qr_{now_ts()}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save QR", sugg, filters)
        if not path:
            return

        root, ext = os.path.splitext(path)
        if not ext:
            path = f"{path}.png"
            ext = ".png"

        ext = ext.lower()
        ext_to_fmt = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".bmp": "BMP",
            ".webp": "WEBP",
            ".tif": "TIFF",
            ".tiff": "TIFF",
            ".ico": "ICO",
        }
        fmt = ext_to_fmt.get(ext, "PNG")

        img = self.current_img
        try:
            # JPEG and BMP do not support alpha; flatten on white
            if fmt in {"JPEG", "BMP"} and img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img_to_save = bg
            elif fmt == "ICO" and img.mode not in ("RGBA", "RGB"):
                img_to_save = img.convert("RGBA")
            else:
                img_to_save = img

            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs.update({"quality": 95, "optimize": True})
            if fmt == "PNG":
                save_kwargs.update({"optimize": True})

            img_to_save.save(path, fmt, **save_kwargs)
            self.status_cb(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Save Image", f"Failed to save: {e}")
            self.status_cb(f"Save error: {e}")

    def copy_png(self):
        if not self.current_img:
            return
        qim = ImageQt.ImageQt(self.current_img)
        QApplication.clipboard().setImage(qim)
        self.status_cb("Copied image")

    def copy_payload(self):
        payload = self.build_payload()
        if payload:
            QApplication.clipboard().setText(payload)
            self.status_cb("Copied payload")

    def import_logo(self):
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.ico);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Logo", "", filters)
        if not path:
            return
        try:
            self.logo_img = Image.open(path)
            self.status_cb("Logo added")
            self.update_preview()
        except Exception as e:
            QMessageBox.warning(self, "Logo", f"Failed to load: {e}")
            self.status_cb(f"Logo load failed: {e}")


# ---------------- Scan Tab ----------------
class ScanTab(QWidget):
    def __init__(self, status_cb):
        super().__init__()
        self.status_cb = status_cb
        self.cap: cv2.VideoCapture | None = None
        self.detector = cv2.QRCodeDetector()
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 fps
        self.timer.timeout.connect(self._on_frame)

        self.open_thread: QThread | None = None
        self.cam_worker: CameraOpener | None = None

        self.seen_recent: dict[str, float] = {}
        self.dup_window = 1.5

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        cam = Card()
        cl = QVBoxLayout(cam)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(8)

        cl.addWidget(HeaderLabel("Camera Feed"))

        self.video = QLabel("Camera off")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumHeight(240)
        self.video.setStyleSheet("border:1px dashed #3a3d44;")
        cl.addWidget(self.video, 1)

        row = QHBoxLayout()
        self.btn_upload = QPushButton("Upload Image"); self.btn_upload.clicked.connect(self.upload_image)
        self.btn_start = QPushButton("Start Camera"); self.btn_start.setObjectName("Primary"); self.btn_start.clicked.connect(self.toggle_camera)
        row.addWidget(self.btn_upload)
        row.addWidget(self.btn_start)
        row.addStretch(1)
        cl.addLayout(row)

        results = Card()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)
        rl.addWidget(HeaderLabel("Results"))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Content", "Type"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        rl.addWidget(self.table)

        root.addWidget(cam, 2)
        root.addWidget(results, 1)

    # Camera control
    def toggle_camera(self):
        if self.cap:
            self._stop_camera()
            return

        self.btn_start.setEnabled(False)
        self.btn_start.setText("Starting…")

        self.open_thread = QThread(self)
        self.cam_worker = CameraOpener(0)
        self.cam_worker.moveToThread(self.open_thread)
        self.cam_worker.opened.connect(self._on_camera_opened)
        self.open_thread.started.connect(self.cam_worker.run)
        self.open_thread.start()

    def _on_camera_opened(self, cap, err):
        try:
            if err or cap is None:
                QMessageBox.warning(self, "Camera", err or "Failed to start camera")
                self.btn_start.setEnabled(True)
                self.btn_start.setText("Start Camera")
                return

            self.cap = cap
            self.timer.start()
            self.btn_start.setEnabled(True)
            self.btn_start.setText("Stop Camera")
            self.status_cb("Camera started")
        finally:
            if self.cam_worker:
                try:
                    self.cam_worker.deleteLater()
                except Exception:
                    pass
            if self.open_thread:
                self.open_thread.quit()
                self.open_thread.wait()
                self.open_thread = None
            self.cam_worker = None

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.video.setText("Camera off")
        self.btn_start.setText("Start Camera")
        self.status_cb("Camera stopped")

    def _on_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        # decode + debounce + add to top
        try:
            data_list: list[str] = []
            retval, infos, points, _ = self.detector.detectAndDecodeMulti(frame)
            if retval:
                for s in infos:
                    if s:
                        data_list.append(s)
            if not data_list:
                s, points, _ = self.detector.detectAndDecode(frame)
                if s:
                    data_list.append(s)

            now = datetime.now().timestamp()
            for s in data_list:
                last = self.seen_recent.get(s, 0.0)
                if now - last >= self.dup_window:
                    self.seen_recent[s] = now
                    self.add_result(s)
        except Exception as e:
            self.status_cb(f"Decode error: {e}")

        qimg = cv_to_qimage(frame)
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def upload_image(self):
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.ico);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", filters)
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback via Pillow for formats not compiled into OpenCV, e.g., WebP
            try:
                pil = Image.open(path)
                if pil.mode in ("RGBA", "LA"):
                    arr = np.array(pil.convert("RGBA"))
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                else:
                    arr = np.array(pil.convert("RGB"))
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception as e:
                QMessageBox.warning(self, "Open Image", f"Failed to open image: {e}")
                return

        self.decode_frame(img)
        qimg = cv_to_qimage(img)
        self.video.setPixmap(
            QPixmap.fromImage(qimg).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def decode_frame(self, frame):
        try:
            data_list: list[str] = []
            retval, infos, points, _ = self.detector.detectAndDecodeMulti(frame)
            if retval:
                for s in infos:
                    if s:
                        data_list.append(s)
            if not data_list:
                s, points, _ = self.detector.detectAndDecode(frame)
                if s:
                    data_list.append(s)
            now = datetime.now().timestamp()
            for s in data_list:
                last = self.seen_recent.get(s, 0.0)
                if now - last >= self.dup_window:
                    self.seen_recent[s] = now
                    self.add_result(s)
        except Exception as e:
            self.status_cb(f"Decode error: {e}")

    def add_result(self, content: str):
        row = 0
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.table.setItem(row, 1, QTableWidgetItem(content))
        low = content.strip().lower()
        if low.startswith(("http://", "https://", "www.")): typ = "URL"
        elif low.startswith("wifi:"):   typ = "Wi-Fi"
        elif low.startswith("smsto:"):  typ = "SMS"
        elif low.startswith("mailto:"): typ = "Email"
        elif low.startswith("begin:vcard"): typ = "vCard"
        elif low.startswith("begin:vcalendar") or low.startswith("begin:vevent"): typ = "Event"
        else: typ = "Text"
        self.table.setItem(row, 2, QTableWidgetItem(typ))
        self.status_cb("QR decoded")


# ---------------- About Tab ----------------
class AboutTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(20)

        # Icon
        pm = QPixmap(80, 80); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(PAL['accent'])); p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, 80, 80, 18, 18)
        p.setBrush(QColor("white"))
        p.drawRect(18, 18, 14, 14); p.drawRect(48, 18, 14, 14); p.drawRect(18, 48, 14, 14)
        p.end()
        icon = QLabel(); icon.setPixmap(pm); icon.setAlignment(Qt.AlignCenter)

        title = QLabel("QR Studio"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:18pt; font-weight:600; color:white;")
        version = QLabel("v1.0.0"); version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"font-size:10pt; color:{PAL['muted']};")

        info = Card(); il = QVBoxLayout(info)
        il.setSpacing(12); il.setContentsMargins(12, 12, 12, 12)

        def section(h, t):
            wrap = QWidget()
            l = QVBoxLayout(wrap); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(4)
            lab = HeaderLabel(h)
            val = QLabel(t); val.setStyleSheet("color:#d1d5db; font-size:10pt;")
            l.addWidget(lab)
            l.addWidget(val)
            return wrap

        il.addWidget(section("Framework", "PySide6 • qrcode • Pillow • OpenCV • numpy"))
        il.addWidget(section("UI Style", "Dark card UI with uniform headers"))
        il.addWidget(section("Fonts", "Tahoma / MS Shell Dlg 2 / Arial"))
        il.addWidget(section("License", "MIT"))
        il.addWidget(section("Shortcuts", "Ctrl+S Save Image • Ctrl+Shift+C Copy Payload • Ctrl+Q Quit • F1 About"))

        root.addWidget(icon)
        root.addWidget(title)
        root.addWidget(version)
        root.addWidget(info)


# ---------------- Main ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR Studio")
        # make window resizable, set sensible minimums
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.statusbar = QStatusBar(); self.setStatusBar(self.statusbar)

        self.gen = GenerateTab(self.status)
        self.scan = ScanTab(self.status)

        self.tabs.addTab(self.gen, "Generate")
        self.tabs.addTab(self.scan, "Scan")
        self.tabs.addTab(AboutTab(), "About")

        # stop camera when leaving Scan tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._apply_styles()
        self._shortcuts()

    def _on_tab_changed(self, idx: int):
        # stop camera if not on Scan tab
        if idx != 1:
            try:
                self.scan._stop_camera()
            except Exception:
                pass

    def status(self, msg: str):
        self.statusbar.showMessage(msg, 3000)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                font-family: "MS Shell Dlg 2", Tahoma, Arial, sans-serif;
                font-size: 10pt;
                color: {PAL['text']};
                background: {PAL['bg']};
            }}
            QFrame#Card {{
                background: {PAL['card']};
                border: 1px solid {PAL['border']};
                border-radius: 6px;
            }}
            QLabel#SectionTitle {{
                font-size: 10pt;
                font-weight: 600;
                color: {PAL['text']};
                background: rgba(255,255,255,0.04);
                border: 1px solid {PAL['border']};
                border-radius: 4px;
                padding-left: 8px;
            }}
            QTextEdit, QLineEdit, QComboBox {{
                background: {PAL['bg']};
                border: 1px solid {PAL['border']};
                border-radius: 4px;
                padding: 6px;
                color: {PAL['text']};
            }}
            QCheckBox {{ color: {PAL['text']}; }}
            QListWidget {{
                background: {PAL['bg']};
                border: 1px solid {PAL['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton {{
                background: {PAL['card']};
                border: 1px solid {PAL['border']};
                border-radius: 6px;
                padding: 6px 12px;
                color: {PAL['text']};
            }}
            QPushButton#Primary {{
                background: {PAL['accent']};
                color: white;
                font-weight: 600;
            }}
            QTableWidget {{
                background: {PAL['bg']};
                border: 1px solid {PAL['border']};
                gridline-color: {PAL['border']};
            }}
        """)

    def _shortcuts(self):
        # scope save to Generate tab
        act_save = QAction(self); act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.gen.save_image); self.addAction(act_save)
        # avoid clobbering text field copy; use Ctrl+Shift+C for payload
        act_copy = QAction(self); act_copy.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act_copy.triggered.connect(self.gen.copy_payload); self.addAction(act_copy)
        act_quit = QAction(self); act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(QApplication.quit); self.addAction(act_quit)
        act_about = QAction(self); act_about.setShortcut(QKeySequence("F1"))
        act_about.triggered.connect(lambda: self.tabs.setCurrentIndex(2)); self.addAction(act_about)

    def closeEvent(self, e):
        try:
            self.scan._stop_camera()
        except Exception:
            pass
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Tahoma", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
