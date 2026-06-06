"""
AI-Generated Image Detector — Desktop App
Phase 4 của luận văn thạc sĩ

Requirements:
    pip install pyqt5 onnxruntime torch torchvision timm opencv-python scikit-image scipy
    pip install git+https://github.com/openai/CLIP.git

Files cần có cùng thư mục với app.py:
    - soft_gating_fusion.onnx
    - model_config.json
"""

import sys, os, json, time
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import local_binary_pattern

import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort

import clip
import timm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui import (
    QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QLinearGradient, QPalette
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH   = os.path.join(BASE_DIR, 'soft_gating_fusion.onnx')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.json')

COLORS = {
    'bg':         '#0D0F14',
    'surface':    '#161820',
    'surface2':   '#1E2028',
    'border':     '#2A2D3A',
    'accent':     '#4F8EF7',
    'accent2':    '#7B5CF0',
    'real':       '#22C55E',
    'fake':       '#EF4444',
    'warn':       '#F59E0B',
    'text':       '#E8EAF0',
    'text_muted': '#6B7280',
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, config_path, device='cpu'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.hc_mean = np.array(self.config['hc_mean'], dtype=np.float32)
        self.hc_std  = np.array(self.config['hc_std'],  dtype=np.float32)
        self.device  = device

        clip_name = self.config.get('clip_model', 'ViT-L/14')
        self.clip_model, self.clip_preprocess = clip.load(clip_name, device=device)
        self.clip_model.eval()

        effnet_name = self.config.get('effnet_model', 'efficientnet_b0')
        self.effnet = timm.create_model(effnet_name, pretrained=True,
                                         num_classes=0, global_pool='avg')
        self.effnet = self.effnet.to(device).eval()

        self.effnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

        self.srm_kernels = np.array([
            [[ 0, 0, 0, 0, 0],[ 0,-1, 2,-1, 0],[ 0, 2,-4, 2, 0],[ 0,-1, 2,-1, 0],[ 0, 0, 0, 0, 0]],
            [[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0],[-1, 2,-2, 2,-1],[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0]],
            [[-1, 0, 0, 0, 1],[ 0,-2, 0, 2, 0],[ 0, 0, 0, 0, 0],[ 0,-2, 0, 2, 0],[-1, 0, 0, 0, 1]]
        ], dtype=np.float32) / 4.0

    def extract_handcrafted(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        srm = np.array([np.abs(ndimage.convolve(gray, k)).mean()
                        for k in self.srm_kernels], dtype=np.float32)

        h, w = gray.shape
        bh, bw = h // 8, w // 8
        gray_c = gray[:bh*8, :bw*8]
        blocks = gray_c.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
        dct_blocks = np.zeros_like(blocks)
        for i in range(bh):
            for j in range(bw):
                dct_blocks[i, j] = cv2.dct(blocks[i, j])
        dct_avg = np.abs(dct_blocks).mean(axis=(0, 1))
        dct = []
        for band in range(15):
            e, c = 0.0, 0
            for u in range(min(band+1, 8)):
                v = band - u
                if v < 8: e += dct_avg[u, v]; c += 1
            dct.append(e / max(c, 1))
        dct = np.array(dct, dtype=np.float32)

        lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        lbp_feat = hist.astype(np.float32)

        img_f = img_np.astype(np.float32) / 255.0
        color = np.array([img_f[:,:,c].mean() for c in range(3)] +
                         [img_f[:,:,c].std()  for c in range(3)], dtype=np.float32)

        lap   = cv2.Laplacian(gray, cv2.CV_32F)
        edges = cv2.Canny(img_np, 50, 150)
        sharp = np.array([lap.var() / 10000.0, edges.mean() / 255.0], dtype=np.float32)

        return np.concatenate([srm, dct, lbp_feat, color, sharp,
                                np.zeros(2, dtype=np.float32)])

    @torch.no_grad()
    def extract(self, pil_img):
        img_np = np.array(pil_img.convert('RGB'))

        img_clip  = self.clip_preprocess(pil_img).unsqueeze(0)
        clip_feat = self.clip_model.encode_image(img_clip).float()
        proj      = self.clip_model.visual.proj
        clip_feat = (clip_feat @ proj.float().T).numpy()

        img_t       = self.effnet_transform(pil_img).unsqueeze(0)
        effnet_feat = self.effnet(img_t).numpy()

        hc_raw  = self.extract_handcrafted(img_np)
        hc_feat = ((hc_raw - self.hc_mean) / (self.hc_std + 1e-8)).reshape(1, -1)

        clip_feat   = clip_feat   / (np.linalg.norm(clip_feat,   axis=1, keepdims=True) + 1e-8)
        effnet_feat = effnet_feat / (np.linalg.norm(effnet_feat, axis=1, keepdims=True) + 1e-8)

        return (clip_feat.astype(np.float32),
                effnet_feat.astype(np.float32),
                hc_feat.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE WORKER
# ─────────────────────────────────────────────────────────────────────────────
class InferenceWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, img_path, extractor, session, config):
        super().__init__()
        self.img_path  = img_path
        self.extractor = extractor
        self.session   = session
        self.config    = config

    def run(self):
        try:
            t0  = time.time()
            img = Image.open(self.img_path).convert('RGB')
            clip_feat, effnet_feat, hc_feat = self.extractor.extract(img)
            score = self.session.run(
                ['score'],
                {'clip_feat': clip_feat, 'effnet_feat': effnet_feat, 'hc_feat': hc_feat}
            )[0][0]
            self.finished.emit({
                'score':   float(score),
                'gates':   self.config['gates'],
                'elapsed': time.time() - t0,
            })
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class ScoreGauge(QWidget):
    def __init__(self):
        super().__init__()
        self.score = None
        self.setFixedSize(200, 200)

    def set_score(self, score):
        self.score = score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h   = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2 - 15

        painter.setPen(QPen(QColor(COLORS['border']), 10))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx-r, cy-r, 2*r, 2*r)

        if self.score is not None:
            color = (QColor(COLORS['real'])  if self.score < 0.3 else
                     QColor(COLORS['fake'])  if self.score > 0.7 else
                     QColor(COLORS['warn']))
            painter.setPen(QPen(color, 10, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx-r, cy-r, 2*r, 2*r, 90*16, -int(self.score * 360 * 16))
            painter.setPen(color)
            painter.setFont(QFont('Consolas', 28, QFont.Bold))
            painter.drawText(QRect(cx-r, cy-25, 2*r, 40), Qt.AlignCenter, f'{self.score:.2f}')
            painter.setPen(QColor(COLORS['text_muted']))
            painter.setFont(QFont('Consolas', 11))
            painter.drawText(QRect(cx-r, cy+20, 2*r, 25), Qt.AlignCenter,
                             'REAL' if self.score < 0.5 else 'FAKE')
        else:
            painter.setPen(QColor(COLORS['text_muted']))
            painter.setFont(QFont('Consolas', 14))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, '—')


class GateBar(QWidget):
    def __init__(self, name, color):
        super().__init__()
        self.name  = name
        self.value = 0.5
        self.color = QColor(color)
        self.setFixedHeight(36)

    def set_value(self, v):
        self.value = float(v)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.setBrush(QBrush(QColor(COLORS['surface2'])))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 6, 6)

        fill_w = max(1, int(self.value * (w - 2)))
        grad   = QLinearGradient(0, 0, fill_w, 0)
        grad.setColorAt(0, self.color.darker(130))
        grad.setColorAt(1, self.color)
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(1, 1, fill_w, h-2, 5, 5)

        painter.setPen(QColor(COLORS['text']))
        painter.setFont(QFont('Consolas', 9, QFont.Bold))
        painter.drawText(QRect(10, 0, w-80, h), Qt.AlignVCenter, self.name)
        painter.setFont(QFont('Consolas', 9))
        painter.drawText(QRect(0, 0, w-10, h), Qt.AlignVCenter | Qt.AlignRight,
                         f'{self.value:.3f}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Image Detector')
        self.setMinimumSize(820, 600)
        self.setAcceptDrops(True)
        self.setStyleSheet(f'background-color: {COLORS["bg"]}; color: {COLORS["text"]};')
        self.worker = None
        self._load_model()
        self._build_ui()

    def _load_model(self):
        try:
            with open(CONFIG_PATH, 'r') as f:
                self.config = json.load(f)
            self.extractor = FeatureExtractor(CONFIG_PATH)
            self.session   = ort.InferenceSession(ONNX_PATH,
                                providers=['CPUExecutionProvider'])
            self.model_ok  = True
        except Exception as e:
            self.model_ok   = False
            self.load_error = str(e)

    def _card(self):
        f = QFrame()
        f.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
            }}
        """)
        return f

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # Header
        header = QLabel('AI IMAGE DETECTOR')
        header.setFont(QFont('Consolas', 17, QFont.Bold))
        header.setStyleSheet(f'color: {COLORS["accent"]}; letter-spacing: 4px;')
        header.setAlignment(Qt.AlignCenter)
        root.addWidget(header)

        sub = QLabel('SoftGating Fusion  ·  Luận văn tốt nghiệp')
        sub.setFont(QFont('Consolas', 9))
        sub.setStyleSheet(f'color: {COLORS["text_muted"]};')
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        # Body
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body)

        # Left: image
        left_card = self._card()
        left_l = QVBoxLayout(left_card)
        left_l.setContentsMargins(16, 16, 16, 16)
        left_l.setSpacing(12)
        body.addWidget(left_card, stretch=3)

        self.img_label = QLabel()
        self.img_label.setFixedSize(360, 360)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(f"""
            background: {COLORS['surface2']};
            border: 2px dashed {COLORS['border']};
            border-radius: 10px;
            color: {COLORS['text_muted']};
            font-family: Consolas; font-size: 12px;
        """)
        self.img_label.setText('Kéo thả ảnh vào đây\nhoặc nhấn nút bên dưới')
        left_l.addWidget(self.img_label)

        self.btn_upload = QPushButton('📁   Chọn ảnh')
        self.btn_upload.setFixedHeight(42)
        self.btn_upload.setFont(QFont('Consolas', 11, QFont.Bold))
        self.btn_upload.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {COLORS['accent']}, stop:1 {COLORS['accent2']});
                color: white; border: none; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #6BA3FF, stop:1 #9B7CF8);
            }}
            QPushButton:disabled {{
                background: {COLORS['border']}; color: {COLORS['text_muted']};
            }}
        """)
        self.btn_upload.clicked.connect(self.open_file)
        left_l.addWidget(self.btn_upload)

        self.status_label = QLabel('Sẵn sàng — chọn ảnh để bắt đầu')
        self.status_label.setFont(QFont('Consolas', 9))
        self.status_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        self.status_label.setAlignment(Qt.AlignCenter)
        left_l.addWidget(self.status_label)

        # Right: results
        right = QVBoxLayout()
        right.setSpacing(12)
        body.addLayout(right, stretch=2)

        score_card = self._card()
        score_l = QVBoxLayout(score_card)
        score_l.setContentsMargins(16, 14, 16, 14)
        score_l.setAlignment(Qt.AlignCenter)

        t = QLabel('FAKE PROBABILITY')
        t.setFont(QFont('Consolas', 9, QFont.Bold))
        t.setStyleSheet(f'color: {COLORS["text_muted"]}; letter-spacing: 2px;')
        t.setAlignment(Qt.AlignCenter)
        score_l.addWidget(t)

        self.gauge = ScoreGauge()
        score_l.addWidget(self.gauge, alignment=Qt.AlignCenter)

        self.time_label = QLabel('')
        self.time_label.setFont(QFont('Consolas', 8))
        self.time_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        self.time_label.setAlignment(Qt.AlignCenter)
        score_l.addWidget(self.time_label)
        right.addWidget(score_card)

        gate_card = self._card()
        gate_l = QVBoxLayout(gate_card)
        gate_l.setContentsMargins(16, 14, 16, 14)
        gate_l.setSpacing(8)

        gt = QLabel('BRANCH WEIGHTS')
        gt.setFont(QFont('Consolas', 9, QFont.Bold))
        gt.setStyleSheet(f'color: {COLORS["text_muted"]}; letter-spacing: 2px;')
        gate_l.addWidget(gt)

        self.bar_clip   = GateBar('CLIP  (1024)', COLORS['accent'])
        self.bar_effnet = GateBar('EfficientNet  (1280)', '#F97316')
        self.bar_hc     = GateBar('Handcrafted  (38)',    COLORS['real'])
        gate_l.addWidget(self.bar_clip)
        gate_l.addWidget(self.bar_effnet)
        gate_l.addWidget(self.bar_hc)
        right.addWidget(gate_card)

        if self.model_ok:
            gates = self.config['gates']
            self.bar_clip.set_value(gates['clip'])
            self.bar_effnet.set_value(gates['effnet'])
            self.bar_hc.set_value(gates['hc'])
            info_text = (f"Val AUC: {self.config.get('seen_auc', 0):.4f}  |  "
                         f"Seen ACC: {self.config.get('seen_acc', 0):.1%}  |  "
                         f"Unseen ACC: {self.config.get('unseen_acc', 0):.1%}")
        else:
            info_text = f'⚠ Model load failed: {self.load_error}'

        info = QLabel(info_text)
        info.setFont(QFont('Consolas', 8))
        info.setStyleSheet(f'color: {COLORS["text_muted"]};')
        info.setAlignment(Qt.AlignCenter)
        info.setWordWrap(True)
        root.addWidget(info)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                self.run_inference(path)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn ảnh', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            self.run_inference(path)

    def run_inference(self, img_path):
        if not self.model_ok:
            self.status_label.setText('⚠ Model chưa load được')
            return
        if self.worker and self.worker.isRunning():
            return

        pixmap = QPixmap(img_path).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.setText('')
        self.status_label.setText('⏳ Đang phân tích...')
        self.btn_upload.setEnabled(False)
        self.gauge.set_score(None)
        self.time_label.setText('')

        self.worker = InferenceWorker(img_path, self.extractor, self.session, self.config)
        self.worker.finished.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_result(self, result):
        score = result['score']
        gates = result['gates']
        self.gauge.set_score(score)
        self.bar_clip.set_value(gates['clip'])
        self.bar_effnet.set_value(gates['effnet'])
        self.bar_hc.set_value(gates['hc'])
        verdict = 'REAL' if score < 0.5 else 'FAKE'
        conf    = (1 - score) if score < 0.5 else score
        self.status_label.setText(f'Kết quả: {verdict}  (confidence {conf:.1%})')
        self.time_label.setText(f'⏱ {result["elapsed"]:.2f}s')
        self.btn_upload.setEnabled(True)

    def on_error(self, msg):
        self.status_label.setText(f'⚠ Lỗi: {msg}')
        self.btn_upload.setEnabled(True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(COLORS['bg']))
    palette.setColor(QPalette.WindowText,      QColor(COLORS['text']))
    palette.setColor(QPalette.Base,            QColor(COLORS['surface']))
    palette.setColor(QPalette.AlternateBase,   QColor(COLORS['surface2']))
    palette.setColor(QPalette.Button,          QColor(COLORS['surface2']))
    palette.setColor(QPalette.ButtonText,      QColor(COLORS['text']))
    palette.setColor(QPalette.Highlight,       QColor(COLORS['accent']))
    palette.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
