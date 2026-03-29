import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QGroupBox,
    QFormLayout, QLabel, QPushButton, QTextEdit, QMessageBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── 활성화 함수 정의 (02_activation_functions.py 독립 재구현) ─────────────────

def sigmoid(x):       return 1 / (1 + np.exp(-x))
def sigmoid_d(x):     s = sigmoid(x); return s * (1 - s)
def tanh_fn(x):       return np.tanh(x)
def tanh_d(x):        return 1 - np.tanh(x) ** 2
def relu(x):          return np.maximum(0, x)
def relu_d(x):        return np.where(x > 0, 1.0, 0.0)
def leaky_relu(x, a=0.01):   return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)


# ── 탭 위젯 ──────────────────────────────────────────────────────────────────

class ActivationTab(QWidget):
    SAVE_FILENAME = '02_activation_functions.png'

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._draw()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 패널 (정적이므로 안내 텍스트만) ─────
        ctrl = QGroupBox('설정')
        ctrl.setFixedWidth(220)
        form = QFormLayout(ctrl)
        form.setSpacing(12)

        info = QLabel(
            '정적 시각화입니다.\n\n'
            '앱 시작 시 자동으로\n'
            '4가지 활성화 함수를\n'
            '비교·렌더링합니다.\n\n'
            '파라미터 조절 없음.'
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.save_btn = QPushButton('그래프 저장')
        self.save_btn.clicked.connect(self.save_figure)
        form.addRow(self.save_btn)

        splitter.addWidget(ctrl)

        # ── 오른쪽: 그래프 ──────────────────────────────
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # ── 하단: 결과 패널 ─────────────────────────────
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        root.addWidget(self.result_text)

    # ── 시각화 ───────────────────────────────────────────────────────────────

    def _draw(self):
        x = np.linspace(-5, 5, 300)
        axes = self.fig.subplots(2, 2)

        # (0,0) 함수 비교
        ax = axes[0, 0]
        ax.plot(x, sigmoid(x),    label='Sigmoid',    linewidth=2)
        ax.plot(x, tanh_fn(x),    label='Tanh',       linewidth=2)
        ax.plot(x, relu(x),       label='ReLU',       linewidth=2)
        ax.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2, linestyle='--')
        ax.axhline(0, color='k', alpha=0.3)
        ax.axvline(0, color='k', alpha=0.3)
        ax.set_title('활성화 함수 비교', fontsize=13, fontweight='bold')
        ax.set_xlabel('입력 x'); ax.set_ylabel('출력 f(x)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # (0,1) 미분 비교
        ax = axes[0, 1]
        ax.plot(x, sigmoid_d(x),    label="Sigmoid'",    linewidth=2)
        ax.plot(x, tanh_d(x),       label="Tanh'",       linewidth=2)
        ax.plot(x, relu_d(x),       label="ReLU'",       linewidth=2)
        ax.plot(x, leaky_relu_d(x), label="Leaky ReLU'", linewidth=2, linestyle='--')
        ax.axhline(0, color='k', alpha=0.3)
        ax.set_title('미분값 (Gradient) 비교', fontsize=13, fontweight='bold')
        ax.set_xlabel('입력 x'); ax.set_ylabel("f'(x)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # (1,0) Sigmoid vs Tanh
        ax = axes[1, 0]
        ax.plot(x, sigmoid(x), label='Sigmoid: 범위 (0,1)', linewidth=3)
        ax.plot(x, tanh_fn(x), label='Tanh: 범위 (-1,1)',   linewidth=3)
        ax.axhline(0,   color='k',    linestyle='-',  alpha=0.3)
        ax.axhline(0.5, color='blue', linestyle='--', alpha=0.3, label='Sigmoid 중심')
        ax.set_title('Sigmoid vs Tanh\n(중심이 다름!)', fontsize=13, fontweight='bold')
        ax.set_xlabel('입력 x'); ax.set_ylabel('출력')
        ax.legend(); ax.grid(True, alpha=0.3)

        # (1,1) ReLU vs Leaky ReLU
        ax = axes[1, 1]
        ax.plot(x, relu(x),       label='ReLU (x<0: 뉴런 죽음)',      linewidth=3)
        ax.plot(x, leaky_relu(x), label='Leaky ReLU (x<0: 살아있음)', linewidth=3)
        ax.axhline(0, color='k', linestyle='-',  alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='ReLU 경계')
        ax.set_title('ReLU vs Leaky ReLU\n(Dying ReLU 문제)', fontsize=13, fontweight='bold')
        ax.set_xlabel('입력 x'); ax.set_ylabel('출력')
        ax.legend(); ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

        self.result_text.setPlainText(
            '■ Sigmoid     범위:(0, 1)    장점:확률 해석 가능          단점:Vanishing Gradient, 0 비중심   용도:이진분류 출력층\n'
            '■ Tanh        범위:(-1, 1)   장점:0 중심, 기울기 큼        단점:Vanishing Gradient 여전히 존재  용도:은닉층, RNN/LSTM\n'
            '■ ReLU        범위:[0, ∞)    장점:계산 빠름, Gradient 소실 없음  단점:Dying ReLU (음수→뉴런 죽음)  용도:현대 신경망 표준\n'
            '■ Leaky ReLU  범위:(-∞, ∞)  장점:Dying ReLU 해결          단점:하이퍼파라미터 α 선택 필요       용도:ReLU 대안\n\n'
            '권장 가이드: 은닉층 → ReLU  |  이진분류 출력층 → Sigmoid  |  회귀 출력층 → 없음(선형)  |  다중분류 출력층 → Softmax'
        )

    # ── 저장 ─────────────────────────────────────────────────────────────────

    def save_figure(self):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, self.SAVE_FILENAME)
        self.fig.savefig(path, dpi=100)
        QMessageBox.information(self, '저장 완료',
                                f'저장 완료:\n{os.path.abspath(path)}')
