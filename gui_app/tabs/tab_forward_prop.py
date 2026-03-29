import os
import numpy as np
import matplotlib.patches as mpatches
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QGroupBox,
    QFormLayout, QLabel, QPushButton, QTextEdit, QMessageBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── 신경망 (03_forward_propagation.py 독립 재구현) ───────────────────────────

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x):    return np.maximum(0, x)


class SimpleNetwork:
    """2-3-1 신경망 (원본과 동일: np.random.seed(42) + np.random.randn)"""
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(2, 3) * 0.5
        self.b1 = np.random.randn(3)    * 0.1
        self.W2 = np.random.randn(3, 1) * 0.5
        self.b2 = np.random.randn(1)    * 0.1

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2


# ── 탭 위젯 ──────────────────────────────────────────────────────────────────

class ForwardPropTab(QWidget):
    SAVE_FILENAME = '03_forward_propagation.png'

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._draw()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 패널 ─────────────────────────────
        ctrl = QGroupBox('설정')
        ctrl.setFixedWidth(220)
        form = QFormLayout(ctrl)
        form.setSpacing(12)

        info = QLabel(
            '정적 시각화입니다.\n\n'
            '고정 입력: [0.5, 0.8]\n'
            '고정 시드: 42\n\n'
            '네트워크 구조: 2-3-1\n'
            '은닉층: ReLU\n'
            '출력층: Sigmoid'
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
        net = SimpleNetwork()
        X   = np.array([0.5, 0.8])
        net.forward(X)

        axes = self.fig.subplots(2, 2)

        # (0,0) 네트워크 구조 다이어그램
        ax = axes[0, 0]
        ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis('off')
        ax.set_title('신경망 구조 (2-3-1)', fontsize=13, fontweight='bold')

        input_y  = [1.2, 2.8]
        hidden_y = [0.5, 2.0, 3.5]

        for i, y in enumerate(input_y):
            c = mpatches.Circle((0.5, y), 0.22, color='#AED6F1', ec='#2980B9', linewidth=2)
            ax.add_patch(c)
            ax.text(0.5, y, f'x{i+1}', ha='center', va='center', fontweight='bold', fontsize=10)

        for i, y in enumerate(hidden_y):
            c = mpatches.Circle((2.0, y), 0.22, color='#A9DFBF', ec='#1E8449', linewidth=2)
            ax.add_patch(c)
            ax.text(2.0, y, f'h{i+1}', ha='center', va='center', fontweight='bold', fontsize=10)

        c = mpatches.Circle((3.5, 2.0), 0.22, color='#F1948A', ec='#C0392B', linewidth=2)
        ax.add_patch(c)
        ax.text(3.5, 2.0, 'y', ha='center', va='center', fontweight='bold', fontsize=10)

        for iy in input_y:
            for hy in hidden_y:
                ax.plot([0.72, 1.78], [iy, hy], 'k-', alpha=0.25, linewidth=1)
        for hy in hidden_y:
            ax.plot([2.22, 3.28], [hy, 2.0], 'k-', alpha=0.25, linewidth=1)

        ax.text(0.5, -0.15, '입력층',       ha='center', fontsize=9, fontweight='bold', color='#2980B9')
        ax.text(2.0, -0.15, '은닉층\n(ReLU)', ha='center', fontsize=9, fontweight='bold', color='#1E8449')
        ax.text(3.5, -0.15, '출력층\n(Sigmoid)', ha='center', fontsize=9, fontweight='bold', color='#C0392B')

        # (0,1) Layer 1 막대 그래프
        ax = axes[0, 1]
        n  = len(net.z1)
        pos = np.arange(n)
        w   = 0.25
        inp_ext = np.zeros(n); inp_ext[:2] = X
        ax.bar(pos - w,  inp_ext,  w, label='입력값',       color='#3498DB', alpha=0.8)
        ax.bar(pos,      net.z1,   w, label='z₁ (ReLU 전)', color='#E67E22', alpha=0.8)
        ax.bar(pos + w,  net.a1,   w, label='a₁ (ReLU 후)', color='#27AE60', alpha=0.8)
        ax.set_title('Layer 1: 입력 → 은닉 (ReLU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('값')
        ax.set_xticks(pos)
        ax.set_xticklabels([f'뉴런 {i+1}' for i in range(n)])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # (1,0) Layer 2 수평 막대
        ax = axes[1, 0]
        layers = ['a₁[0] (은닉 입력)', 'z₂ (Sigmoid 전)', 'a₂ (Sigmoid 후)']
        vals   = [net.a1[0], net.z2[0], net.a2[0]]
        colors = ['#27AE60', '#E67E22', '#E74C3C']
        for v, lbl, clr in zip(vals, layers, colors):
            ax.barh(lbl, v, color=clr, alpha=0.8)
            ax.text(v + 0.02, lbl, f'{v:.4f}', va='center', fontsize=10)
        ax.set_title('Layer 2: 은닉 → 출력 (Sigmoid)', fontsize=12, fontweight='bold')
        ax.set_xlabel('값'); ax.grid(True, alpha=0.3)

        # (1,1) 수식 요약
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('행렬 연산 수식', fontsize=13, fontweight='bold')
        lines = [
            'Forward Propagation:',
            '',
            'Layer 1 (입력 → 은닉):',
            f'  z₁ = X @ W₁ + b₁',
            f'  z₁ = {np.round(net.z1, 3)}',
            f'  a₁ = ReLU(z₁)',
            f'  a₁ = {np.round(net.a1, 3)}',
            '',
            'Layer 2 (은닉 → 출력):',
            f'  z₂ = a₁ @ W₂ + b₂ = {net.z2[0]:.4f}',
            f'  a₂ = Sigmoid(z₂)  = {net.a2[0]:.4f}',
            '',
            f'  ★ 최종 출력: {net.a2[0]:.4f}',
        ]
        y_pos = 0.95
        for line in lines:
            if line == '':
                y_pos -= 0.04
            else:
                bold = line.endswith(':') or line.startswith('Forward') or '★' in line
                ax.text(0.03, y_pos, line,
                        fontsize=9, fontweight='bold' if bold else 'normal',
                        family='monospace', transform=ax.transAxes)
                y_pos -= 0.068

        self.fig.tight_layout()
        self.canvas.draw()

        # 결과 패널
        self.result_text.setPlainText(
            f'[순전파 계산 결과]  입력: [{X[0]}, {X[1]}]  /  네트워크: 2-3-1\n\n'
            f'  Layer 1\n'
            f'    z₁ (선형 결합, ReLU 전)  = {np.round(net.z1, 5)}\n'
            f'    a₁ (ReLU 활성화 후)      = {np.round(net.a1, 5)}\n\n'
            f'  Layer 2\n'
            f'    z₂ (선형 결합, Sigmoid 전) = {net.z2[0]:.6f}\n'
            f'    a₂ (Sigmoid 활성화 후)     = {net.a2[0]:.6f}  ← 최종 출력\n\n'
            f'  관찰: ReLU 이후 음수 값이 0으로 클리핑됨  /  Sigmoid 이후 0~1 범위로 압축됨'
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
