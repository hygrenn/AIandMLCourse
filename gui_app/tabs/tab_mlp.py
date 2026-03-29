import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QGroupBox,
    QFormLayout, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── MLP 로직 (04_mlp_numpy.py 독립 재구현) ───────────────────────────────────

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _sigmoid_d(x):
    s = _sigmoid(x)
    return s * (1 - s)


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.W1 = np.random.randn(input_size, hidden_size)  * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr  = lr
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = _sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = _sigmoid(self.z2)
        return self.a2

    def _backward(self, X, y, out):
        m   = X.shape[0]
        dz2 = out - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = dz2.sum(axis=0, keepdims=True) / m
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _sigmoid_d(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = dz1.sum(axis=0, keepdims=True) / m
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def train(self, X, y, epochs, progress_cb=None):
        step = max(1, epochs // 100)
        for epoch in range(epochs):
            out  = self.forward(X)
            loss = float(np.mean((out - y) ** 2))
            self.loss_history.append(loss)
            self._backward(X, y, out)
            if progress_cb and epoch % step == 0:
                progress_cb(int(epoch / epochs * 100))

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ── 학습 워커 (QThread) ────────────────────────────────────────────────────────

class MLPWorker(QThread):
    finished = Signal(dict)
    progress = Signal(int)

    def __init__(self, hidden_size, lr, epochs):
        super().__init__()
        self.hidden_size = hidden_size
        self.lr          = lr
        self.epochs      = epochs

    def run(self):
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]],          dtype=float)

        mlp = MLP(2, self.hidden_size, 1, self.lr)
        mlp.train(X, y, self.epochs, progress_cb=lambda p: self.progress.emit(p))

        preds       = mlp.forward(X)
        pred_labels = mlp.predict(X)
        accuracy    = float(np.mean(pred_labels == y.astype(int)) * 100)

        # 결정 경계 그리드
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                             np.linspace(-0.5, 1.5, 200))
        Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        self.finished.emit({
            'loss_history':       mlp.loss_history,
            'predictions':        preds,
            'pred_labels':        pred_labels,
            'accuracy':           accuracy,
            'hidden_activations': mlp.a1,
            'X': X, 'y': y,
            'xx': xx, 'yy': yy, 'Z': Z,
        })


# ── 탭 위젯 ──────────────────────────────────────────────────────────────────

class MLPTab(QWidget):
    SAVE_FILENAME = '04_mlp_training.png'

    def __init__(self):
        super().__init__()
        self.worker = None
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

        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(2, 32)
        self.hidden_spin.setValue(4)
        form.addRow('은닉층 뉴런 수:', self.hidden_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 2.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(0.5)
        self.lr_spin.setDecimals(2)
        form.addRow('학습률:', self.lr_spin)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1000, 50000)
        self.epoch_spin.setSingleStep(1000)
        self.epoch_spin.setValue(10000)
        form.addRow('에폭 수:', self.epoch_spin)

        self.train_btn = QPushButton('학습 시작')
        self.train_btn.clicked.connect(self.start_training)
        form.addRow(self.train_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        form.addRow(self.progress_bar)

        self.save_btn = QPushButton('그래프 저장')
        self.save_btn.clicked.connect(self.save_figure)
        self.save_btn.setEnabled(False)
        form.addRow(self.save_btn)

        splitter.addWidget(ctrl)

        # ── 오른쪽: 그래프 ──────────────────────────────
        self.fig = Figure(figsize=(14, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # ── 하단: 결과 패널 ─────────────────────────────
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText(
            '학습 시작 버튼을 눌러 XOR 학습을 실행하세요.\n'
            '(에폭 10,000 기준 약 2~5초 소요)'
        )
        root.addWidget(self.result_text)

    # ── 학습 제어 ─────────────────────────────────────────────────────────────

    def start_training(self):
        self.train_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.setPlainText('학습 중...')

        self.worker = MLPWorker(
            self.hidden_spin.value(),
            self.lr_spin.value(),
            self.epoch_spin.value(),
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _on_done(self, r):
        self.progress_bar.setValue(100)
        self.train_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._draw(r)
        self._update_result(r)

    # ── 시각화 ───────────────────────────────────────────────────────────────

    def _draw(self, r):
        self.fig.clf()
        axes = self.fig.subplots(1, 3)

        # (1) 손실 그래프
        ax = axes[0]
        ax.plot(r['loss_history'], linewidth=2, color='#2980B9')
        ax.set_title('학습 손실 (MSE)', fontsize=13, fontweight='bold')
        ax.set_xlabel('에폭'); ax.set_ylabel('손실 (log scale)')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        # (2) 결정 경계
        ax = axes[1]
        cf = ax.contourf(r['xx'], r['yy'], r['Z'], levels=20,
                         cmap='RdYlBu', alpha=0.85)
        self.fig.colorbar(cf, ax=ax, label='출력 확률')
        for pt, lbl in zip(r['X'], r['y']):
            clr = '#CC0000' if lbl[0] == 1 else '#0000CC'
            mk  = 'o'       if lbl[0] == 1 else 'x'
            kw  = dict(edgecolors='black', linewidth=3) if mk == 'o' else dict(linewidth=3)
            ax.scatter(pt[0], pt[1], c=clr, marker=mk, s=300, zorder=5, **kw)
            ax.text(pt[0], pt[1] - 0.16,
                    f'({int(pt[0])},{int(pt[1])})',
                    ha='center', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_title('XOR 결정 경계', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # (3) 은닉층 활성화
        ax    = axes[2]
        ha    = r['hidden_activations']
        n_h   = ha.shape[1]
        im    = ax.imshow(ha.T, cmap='viridis', aspect='auto',
                          vmin=0, vmax=1)
        ax.set_yticks(range(n_h))
        ax.set_yticklabels([f'은닉 {i+1}' for i in range(n_h)], fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'], fontsize=9)
        ax.set_title('은닉층 활성화 값', fontsize=13, fontweight='bold')
        ax.set_xlabel('입력 패턴')
        self.fig.colorbar(im, ax=ax, label='활성화')
        for i in range(n_h):
            for j in range(4):
                ax.text(j, i, f'{ha[j, i]:.2f}',
                        ha='center', va='center',
                        color='white', fontweight='bold', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_result(self, r):
        X, y  = r['X'], r['y']
        preds = r['predictions']
        lines = [
            f'최종 손실 (MSE): {r["loss_history"][-1]:.6f}',
            f'정확도: {r["accuracy"]:.1f}%',
            '',
            '입력       예측값    정답',
            '─' * 32,
        ]
        for inp, pred, lbl in zip(X, preds, y):
            ok = '✓' if pred[0] == int(lbl[0]) else '✗'
            lines.append(
                f'  ({int(inp[0])},{int(inp[1])})  →  {pred[0]:.4f}   |  {int(lbl[0])}  {ok}'
            )
        if r['accuracy'] == 100.0:
            lines.append('\n→ XOR 문제 해결 성공! Multi-Layer Perceptron의 힘!')
        else:
            lines.append(f'\n→ 정확도 {r["accuracy"]:.1f}% — 에폭/학습률을 조정해 보세요.')
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
