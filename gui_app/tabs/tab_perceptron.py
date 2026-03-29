import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QGroupBox,
    QFormLayout, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QMessageBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── 퍼셉트론 로직 (01_perceptron.py에서 독립 재구현) ──────────────────────────

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # 원본과 동일: 매 실행마다 랜덤 초기화 (seed 없음)
        self.weights = np.random.randn(input_size)
        self.bias    = float(np.random.randn())
        self.lr      = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train(self, X, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(X, labels):
                err = label - self.predict(inputs)
                self.weights += self.lr * err * inputs
                self.bias    += self.lr * err


# ── 탭 위젯 ──────────────────────────────────────────────────────────────────

class PerceptronTab(QWidget):
    SAVE_FILENAME = '01_perceptron.png'

    def __init__(self):
        super().__init__()
        self.fig = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 패널 ─────────────────────────────
        ctrl = QGroupBox('설정')
        ctrl.setFixedWidth(220)
        form = QFormLayout(ctrl)
        form.setSpacing(10)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(0.1)
        self.lr_spin.setDecimals(2)
        form.addRow('학습률:', self.lr_spin)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 1000)
        self.epoch_spin.setSingleStep(10)
        self.epoch_spin.setValue(100)
        form.addRow('에폭 수:', self.epoch_spin)

        self.train_btn = QPushButton('학습 시작')
        self.train_btn.clicked.connect(self.run_training)
        form.addRow(self.train_btn)

        self.save_btn = QPushButton('그래프 저장')
        self.save_btn.clicked.connect(self.save_figure)
        self.save_btn.setEnabled(False)
        form.addRow(self.save_btn)

        splitter.addWidget(ctrl)

        # ── 오른쪽: 그래프 ──────────────────────────────
        self.fig = Figure(figsize=(13, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # ── 하단: 결과 패널 ─────────────────────────────
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText(
            '학습 시작 버튼을 눌러 AND / OR / XOR 게이트 학습 결과를 확인하세요.'
        )
        root.addWidget(self.result_text)

    # ── 학습 ─────────────────────────────────────────────────────────────────

    def run_training(self):
        lr     = self.lr_spin.value()
        epochs = self.epoch_spin.value()

        X     = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y_and = np.array([0, 0, 0, 1])
        y_or  = np.array([0, 1, 1, 1])
        y_xor = np.array([0, 1, 1, 0])

        p_and = Perceptron(2, lr)
        p_and.train(X, y_and, epochs)

        p_or  = Perceptron(2, lr)
        p_or.train(X, y_or, epochs)

        # XOR은 더 많은 에폭 필요 (단일 퍼셉트론으로 해결 불가)
        p_xor = Perceptron(2, lr)
        p_xor.train(X, y_xor, epochs * 10)

        self._draw(p_and, p_or, p_xor, X, y_and, y_or, y_xor)
        self._update_result(p_and, p_or, p_xor, X, y_and, y_or, y_xor)
        self.save_btn.setEnabled(True)

    # ── 시각화 ───────────────────────────────────────────────────────────────

    def _draw(self, p_and, p_or, p_xor, X, y_and, y_or, y_xor):
        self.fig.clf()
        axes = self.fig.subplots(1, 3)
        configs = [
            (axes[0], p_and, y_and, 'AND Gate\n(선형 분리 가능)'),
            (axes[1], p_or,  y_or,  'OR Gate\n(선형 분리 가능)'),
            (axes[2], p_xor, y_xor, 'XOR Gate\n(선형 분리 불가능!)'),
        ]
        for ax, p, y, title in configs:
            self._plot_boundary(ax, p, X, y, title)
        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _plot_boundary(ax, perceptron, X, y, title):
        x_min, x_max = -0.5, 1.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(x_min, x_max, 150),
        )
        Z = np.array([
            perceptron.predict(np.array([xi, yi]))
            for xi, yi in zip(xx.ravel(), yy.ravel())
        ]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3,
                    levels=[-0.5, 0.5, 1.5], colors=['#4488FF', '#FF4444'])
        for point, label in zip(X, y):
            color  = '#CC0000' if label == 1 else '#0000CC'
            marker = 'o'       if label == 1 else 'x'
            ax.scatter(point[0], point[1], c=color, marker=marker,
                       s=200, edgecolors='black', linewidth=2, zorder=3)

        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)

    # ── 결과 텍스트 ──────────────────────────────────────────────────────────

    def _update_result(self, p_and, p_or, p_xor, X, y_and, y_or, y_xor):
        lines = []
        for name, p, y in [('AND', p_and, y_and),
                            ('OR',  p_or,  y_or),
                            ('XOR', p_xor, y_xor)]:
            errors = sum(p.predict(inp) != lbl for inp, lbl in zip(X, y))
            lines.append(f'[{name} 게이트]  오류: {errors}/4')
            for inp, lbl in zip(X, y):
                pred = p.predict(inp)
                mark = '✓' if pred == lbl else '✗'
                lines.append(f'  입력({int(inp[0])},{int(inp[1])}) → 예측:{pred}  정답:{lbl}  {mark}')
            lines.append('')
        lines.append('→ XOR은 단일 퍼셉트론으로 해결 불가능! (Multi-Layer 필요)')
        self.result_text.setPlainText('\n'.join(lines))

    # ── 저장 ─────────────────────────────────────────────────────────────────

    def save_figure(self):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, self.SAVE_FILENAME)
        self.fig.savefig(path, dpi=100)
        QMessageBox.information(self, '저장 완료',
                                f'저장 완료:\n{os.path.abspath(path)}')
