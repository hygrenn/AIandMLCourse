import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QLabel,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── 활성화 함수 ───────────────────────────────────────────────────────────────

def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def _relu(x):    return np.maximum(0, x)
def _tanh(x):    return np.tanh(x)


# ── 근사할 타깃 함수 ─────────────────────────────────────────────────────────

def target_sin(x):     return np.sin(2 * np.pi * x)
def target_step(x):    return np.where(x < 0.5, 0.0, 1.0)
def target_complex(x): return (np.sin(2*np.pi*x)
                               + 0.5*np.sin(4*np.pi*x)
                               + 0.3*np.cos(6*np.pi*x))


# ── UniversalApproximator (05_universal_approximation.py 독립 재구현) ─────────

class UniversalApproximator:
    def __init__(self, n_hidden, activation='tanh'):
        self.n_hidden   = n_hidden
        self.activation = activation

        lim1 = np.sqrt(6 / (1 + n_hidden))
        self.W1 = np.random.uniform(-lim1, lim1, (1, n_hidden))
        self.b1 = np.zeros(n_hidden)

        lim2 = np.sqrt(6 / (n_hidden + 1))
        self.W2 = np.random.uniform(-lim2, lim2, (n_hidden, 1))
        self.b2 = np.zeros(1)

    def _act(self, z):
        if self.activation == 'tanh':    return _tanh(z)
        elif self.activation == 'relu':  return _relu(z)
        else:                            return _sigmoid(z)

    def _act_d(self, z, a):
        if self.activation == 'tanh':    return 1 - a ** 2
        elif self.activation == 'relu':  return (z > 0).astype(float)
        else:                            return a * (1 - a)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self._act(z1)
        return a1 @ self.W2 + self.b2

    def train(self, X, y, epochs=5000, lr=0.01):
        for _ in range(epochs):
            z1  = X @ self.W1 + self.b1
            a1  = self._act(z1)
            out = a1 @ self.W2 + self.b2

            dout  = 2 * (out - y) / len(X)
            dW2   = a1.T @ dout
            db2   = dout.sum(axis=0)
            da1   = dout @ self.W2.T
            dz1   = da1 * self._act_d(z1, a1)
            dW1   = X.T @ dz1
            db1   = dz1.sum(axis=0)

            self.W2 -= lr * dW2;  self.b2 -= lr * db2
            self.W1 -= lr * dW1;  self.b1 -= lr * db1


# ── 학습 워커 (QThread) ────────────────────────────────────────────────────────

class UniversalWorker(QThread):
    finished = Signal(list)
    progress = Signal(int, str)   # (퍼센트, 상태 메시지)

    def __init__(self, activation, epochs):
        super().__init__()
        self.activation = activation
        self.epochs     = epochs

    def run(self):
        x_train = np.linspace(0, 1, 100).reshape(-1, 1)
        x_test  = np.linspace(0, 1, 200).reshape(-1, 1)

        targets       = [('사인파', target_sin),
                         ('계단 함수', target_step),
                         ('복합 함수', target_complex)]
        neuron_counts = [3, 10, 50]

        results = []
        total   = len(targets) * len(neuron_counts)
        idx     = 0

        for title, func in targets:
            y_train = func(x_train)
            y_true  = func(x_test)
            for n in neuron_counts:
                idx += 1
                self.progress.emit(
                    int(idx / total * 100),
                    f'{title} / {n}개 뉴런 학습 중… ({idx}/{total})'
                )
                lr = 0.05 if n < 20 else 0.01
                model = UniversalApproximator(n, self.activation)
                model.train(x_train, y_train, self.epochs, lr)
                y_pred = model.forward(x_test)
                mse    = float(np.mean((y_pred - y_true) ** 2))
                results.append({
                    'title':    title,
                    'n_neurons': n,
                    'x_test':   x_test,
                    'y_true':   y_true,
                    'y_pred':   y_pred,
                    'x_train':  x_train[::10],
                    'y_train':  y_train[::10],
                    'mse':      mse,
                })

        self.finished.emit(results)


# ── 탭 위젯 ──────────────────────────────────────────────────────────────────

class UniversalTab(QWidget):
    SAVE_FILENAME = '05_universal_approximation.png'

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

        self.act_combo = QComboBox()
        self.act_combo.addItems(['tanh', 'relu', 'sigmoid'])
        form.addRow('활성화 함수:', self.act_combo)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1000, 10000)
        self.epoch_spin.setSingleStep(1000)
        self.epoch_spin.setValue(5000)
        form.addRow('에폭 수:', self.epoch_spin)

        self.train_btn = QPushButton('학습 시작')
        self.train_btn.clicked.connect(self.start_training)
        form.addRow(self.train_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        form.addRow(self.progress_bar)

        self.status_lbl = QLabel('')
        self.status_lbl.setWordWrap(True)
        form.addRow(self.status_lbl)

        self.save_btn = QPushButton('그래프 저장')
        self.save_btn.clicked.connect(self.save_figure)
        self.save_btn.setEnabled(False)
        form.addRow(self.save_btn)

        splitter.addWidget(ctrl)

        # ── 오른쪽: 그래프 ──────────────────────────────
        self.fig = Figure(figsize=(14, 10))
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # ── 하단: 결과 패널 ─────────────────────────────
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText(
            '학습 시작 버튼을 눌러 보편 근사 정리를 시연하세요.\n'
            '(9개 모델 순차 학습 — 약 20~40초 소요)'
        )
        root.addWidget(self.result_text)

    # ── 학습 제어 ─────────────────────────────────────────────────────────────

    def start_training(self):
        self.train_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_lbl.setText('준비 중…')
        self.result_text.setPlainText('학습 중… 잠시 기다려 주세요.')

        self.worker = UniversalWorker(
            self.act_combo.currentText(),
            self.epoch_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(msg)

    def _on_done(self, results):
        self.progress_bar.setValue(100)
        self.status_lbl.setText('학습 완료!')
        self.train_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._draw(results)
        self._update_result(results)

    # ── 시각화 ───────────────────────────────────────────────────────────────

    def _draw(self, results):
        self.fig.clf()
        titles   = ['사인파', '계단 함수', '복합 함수']
        neurons  = [3, 10, 50]
        lookup   = {(r['title'], r['n_neurons']): r for r in results}
        axes     = self.fig.subplots(3, 3)

        for row, n in enumerate(neurons):
            for col, title in enumerate(titles):
                ax = axes[row, col]
                r  = lookup[(title, n)]
                ax.plot(r['x_test'], r['y_true'],
                        'b-', linewidth=2, label='실제 함수', alpha=0.75)
                ax.plot(r['x_test'], r['y_pred'],
                        'r--', linewidth=2, label=f'신경망 ({n}뉴런)')
                ax.scatter(r['x_train'], r['y_train'],
                           c='green', s=20, alpha=0.5, label='학습 데이터')
                t = (f'{title}\n{n}뉴런 (MSE:{r["mse"]:.4f})'
                     if row == 0 else f'{n}뉴런  MSE:{r["mse"]:.4f}')
                ax.set_title(t, fontsize=9 + (1 if row == 0 else 0),
                             fontweight='bold' if row == 0 else 'normal')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x'); ax.set_ylabel('y')

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def _update_result(self, results):
        titles  = ['사인파', '계단 함수', '복합 함수']
        neurons = [3, 10, 50]
        lookup  = {(r['title'], r['n_neurons']): r for r in results}

        lines = ['[보편 근사 MSE 결과표]', '']
        header = f'{"뉴런 수":>8}  ' + '  '.join(f'{t:>10}' for t in titles)
        lines.append(header)
        lines.append('─' * (len(header) + 4))
        for n in neurons:
            row = f'{str(n)+"개":>8}  '
            for t in titles:
                row += f'{lookup[(t,n)]["mse"]:>10.6f}  '
            lines.append(row)
        lines += [
            '',
            '관찰:',
            '  3개 뉴런  → 매우 거친 근사 (MSE 높음)',
            ' 10개 뉴런  → 대략적 형태',
            ' 50개 뉴런  → 거의 완벽 (Universal Approximation Theorem 확인!)',
        ]
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
