import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from tensorflow import keras

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QMessageBox, QToolButton,
)

from core.models import build_lab1_model, build_lab1_size_models, build_lab1_extreme_model
from core.trainer import TrainingWorker, make_keras_callback

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

_FUNCTIONS = {
    'sin(x)':          np.sin,
    'cos(x)+0.5sin(2x)': lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
    'x·sin(x)':        lambda x: x * np.sin(x),
    'tanh(x)':         np.tanh,
    'x³':              lambda x: x ** 3,
}

_EXTREME_FN = (
    lambda x: np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.cos(3 * x)
              + 0.2 * np.sin(5 * x) + 0.1 * x * np.cos(x)
)


def generate_lab1_data(func_name, n_points=500, x_range=(-3.0, 3.0)):
    """Shuffled (X, y) for 1D function approximation. X shape: (n,1), y shape: (n,1)."""
    if func_name not in _FUNCTIONS:
        raise ValueError(f"Unknown function: '{func_name}'. Choose from {list(_FUNCTIONS.keys())}")
    x = np.linspace(*x_range, n_points)
    x = x[np.random.permutation(len(x))]
    y = _FUNCTIONS[func_name](x)
    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def lab1_experiment(params, should_stop, emit_progress):
    """
    3개 실험 순차 실행.
    Returns dict with keys: 'basic', 'size_comparison', 'extreme', optionally 'advanced'.
    """
    epochs = params['epochs']
    lr = params['learning_rate']
    func_name = params['function']
    advanced = params.get('advanced', False)
    results = {}

    # ① 기본 함수 근사
    X, y = generate_lab1_data(func_name)
    m = build_lab1_model()
    m.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    cb = make_keras_callback(f'기본 ({func_name})', epochs, should_stop, emit_progress)
    h = m.fit(X, y, epochs=epochs, validation_split=0.2, callbacks=[cb], verbose=0)
    results['basic'] = {'model': m, 'history': h, 'X': X, 'y': y, 'func': func_name}
    if should_stop():
        return results

    # ② 네트워크 크기 비교 (고정 1000 epoch)
    SIZE_EPOCHS = 1000
    size_results = {}
    for name, sm in build_lab1_size_models().items():
        sm.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
        cb = make_keras_callback(f'크기 비교 ({name})', SIZE_EPOCHS, should_stop, emit_progress)
        sh = sm.fit(X, y, epochs=SIZE_EPOCHS, callbacks=[cb], verbose=0)
        size_results[name] = {'model': sm, 'history': sh}
        if should_stop():
            break
    results['size_comparison'] = size_results
    if should_stop():
        return results

    # ③ 극한 복잡도 테스트 (고정 2000 epoch)
    EXTREME_EPOCHS = 2000
    x_ex = np.linspace(-3, 3, 500)
    x_ex = x_ex[np.random.permutation(len(x_ex))].astype(np.float32)
    X_ex = x_ex.reshape(-1, 1)
    y_ex = _EXTREME_FN(X_ex)
    em = build_lab1_extreme_model()
    em.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    cb = make_keras_callback('극한 복잡도', EXTREME_EPOCHS, should_stop, emit_progress)
    eh = em.fit(X_ex, y_ex, epochs=EXTREME_EPOCHS, validation_split=0.2, callbacks=[cb], verbose=0)
    results['extreme'] = {'model': em, 'history': eh, 'X': X_ex, 'y': y_ex}
    if should_stop():
        return results

    # ④ 심화: tanh(x), x³
    if advanced:
        x_plot_adv = np.linspace(-3, 3, 300).reshape(-1, 1).astype(np.float32)
        adv = {}
        for fn_name in ['tanh(x)', 'x³']:
            X_a, y_a = generate_lab1_data(fn_name)
            am = build_lab1_model()
            am.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
            cb = make_keras_callback(f'심화 ({fn_name})', epochs, should_stop, emit_progress)
            ah = am.fit(X_a, y_a, epochs=epochs, validation_split=0.2, callbacks=[cb], verbose=0)
            adv[fn_name] = {
                'history': ah,
                'x_plot': x_plot_adv,
                'y_true': _FUNCTIONS[fn_name](x_plot_adv),
                'y_pred': am.predict(x_plot_adv, verbose=0),
                'final_loss': ah.history['loss'][-1],
            }
            if should_stop():
                break
        results['advanced'] = adv

    return results


def _tooltip_btn(text):
    b = QToolButton()
    b.setText('?')
    b.setToolTip(text)
    b.setFixedSize(18, 18)
    b.setStyleSheet(
        'QToolButton { color: #7fb8ff; font-size: 10px;'
        'border: 1px solid #555; border-radius: 9px; background: transparent; }'
    )
    return b


class Lab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._results = None
        self._figure = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Parameter panel ──────────────────────────────────
        pp = QWidget()
        pp.setFixedWidth(210)
        pp.setStyleSheet('background: #1a1a2e; border-right: 1px solid #333;')
        pl = QVBoxLayout(pp)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(8)

        hdr = QLabel('파라미터')
        hdr.setStyleSheet(
            'color: #ccc; font-weight: bold; font-size: 11px;'
            'border-bottom: 1px solid #333; padding-bottom: 4px;'
        )
        pl.addWidget(hdr)

        lab_hdr = QLabel('Lab 1: 1D 함수 근사')
        lab_hdr.setStyleSheet('color: #7fb8ff; font-size: 10px;')
        pl.addWidget(lab_hdr)

        # Function selector
        r = QHBoxLayout()
        lbl = QLabel('함수 선택')
        lbl.setStyleSheet('color: #aaa; font-size: 10px;')
        r.addWidget(lbl)
        r.addWidget(_tooltip_btn('근사할 1D 함수를 선택합니다.'))
        pl.addLayout(r)
        self._func_combo = QComboBox()
        self._func_combo.addItems(['sin(x)', 'cos(x)+0.5sin(2x)', 'x·sin(x)'])
        self._func_combo.setStyleSheet(
            'QComboBox { background: #2a2a3e; color: #ccc; font-size: 10px;'
            'border: 1px solid #444; padding: 3px; }'
        )
        pl.addWidget(self._func_combo)

        # Epochs
        r2 = QHBoxLayout()
        l2 = QLabel('Epochs')
        l2.setStyleSheet('color: #aaa; font-size: 10px;')
        r2.addWidget(l2)
        r2.addWidget(_tooltip_btn(
            '기본 함수 근사 에포크 수.\n크기 비교: 1000 에포크 고정\n극한 테스트: 2000 에포크 고정'
        ))
        pl.addLayout(r2)
        self._epochs_input = QLineEdit('3000')
        self._epochs_input.setStyleSheet(
            'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
            'border: 1px solid #444; padding: 3px; }'
        )
        pl.addWidget(self._epochs_input)

        # Learning rate
        r3 = QHBoxLayout()
        l3 = QLabel('Learning Rate')
        l3.setStyleSheet('color: #aaa; font-size: 10px;')
        r3.addWidget(l3)
        r3.addWidget(_tooltip_btn('Adam optimizer learning rate.\n권장: 0.0001 ~ 0.01'))
        pl.addLayout(r3)
        self._lr_input = QLineEdit('0.001')
        self._lr_input.setStyleSheet(
            'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
            'border: 1px solid #444; padding: 3px; }'
        )
        pl.addWidget(self._lr_input)

        # Advanced
        adv_lbl = QLabel('심화 과제')
        adv_lbl.setStyleSheet(
            'color: #7fb8ff; font-size: 10px; font-weight: bold;'
            'border-top: 1px solid #333; padding-top: 8px; margin-top: 4px;'
        )
        pl.addWidget(adv_lbl)
        self._advanced_check = QCheckBox('tanh(x), x³ 추가 학습')
        self._advanced_check.setStyleSheet('color: #aaa; font-size: 10px;')
        pl.addWidget(self._advanced_check)

        pl.addStretch()

        self._start_btn = QPushButton('▶  학습 시작')
        self._start_btn.setStyleSheet(
            'QPushButton { background: #4a7a4a; color: #fff; border: none;'
            'padding: 6px; border-radius: 4px; font-size: 11px; }'
            'QPushButton:hover { background: #5a8a5a; }'
            'QPushButton:disabled { background: #333; color: #666; }'
        )
        self._start_btn.clicked.connect(self._start)
        pl.addWidget(self._start_btn)

        self._stop_btn = QPushButton('■  중단')
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            'QPushButton { background: #7a4a4a; color: #fff; border: none;'
            'padding: 6px; border-radius: 4px; font-size: 11px; }'
            'QPushButton:hover { background: #8a5a5a; }'
            'QPushButton:disabled { background: #333; color: #666; }'
        )
        self._stop_btn.clicked.connect(self._stop)
        pl.addWidget(self._stop_btn)

        root.addWidget(pp)

        # ── Graph panel ──────────────────────────────────────
        gp = QWidget()
        gl = QVBoxLayout(gp)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(0)

        plt.style.use('dark_background')
        self._figure, self._axes = plt.subplots(1, 3, figsize=(13, 4))
        self._figure.patch.set_facecolor('#12121e')
        for ax in self._axes:
            ax.set_facecolor('#12121e')
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, gp)
        self._toolbar.setStyleSheet('background: #1a1a2e; color: #aaa; border: none;')
        gl.addWidget(self._canvas)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(8, 4, 8, 4)
        bottom.addWidget(self._toolbar)
        bottom.addStretch()
        self._save_btn = QPushButton('💾  PNG 저장')
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet(
            'QPushButton { background: #333; color: #aaa; border: 1px solid #444;'
            'padding: 3px 10px; border-radius: 3px; font-size: 10px; }'
            'QPushButton:hover { background: #444; }'
            'QPushButton:disabled { color: #555; }'
        )
        self._save_btn.clicked.connect(self._save)
        bottom.addWidget(self._save_btn)
        gl.addLayout(bottom)

        root.addWidget(gp)

    def get_params(self):
        try:
            epochs = int(self._epochs_input.text())
        except ValueError:
            raise ValueError('Epochs는 정수여야 합니다.')
        if epochs < 1:
            raise ValueError('Epochs는 1 이상이어야 합니다.')
        try:
            lr = float(self._lr_input.text())
        except ValueError:
            raise ValueError('Learning Rate는 숫자여야 합니다.')
        if lr <= 0:
            raise ValueError('Learning Rate는 0보다 커야 합니다.')
        return {
            'epochs': epochs,
            'learning_rate': lr,
            'function': self._func_combo.currentText(),
            'advanced': self._advanced_check.isChecked(),
        }

    def _start(self):
        try:
            params = self.get_params()
        except ValueError as e:
            QMessageBox.warning(self, '입력 오류', str(e))
            return
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._worker = TrainingWorker(lab1_experiment, params)
        mw = self.window()
        if hasattr(mw, 'update_status'):
            self._worker.progress.connect(mw.update_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)

    def _on_finished(self, results):
        self._results = results
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(True)
        self.render_results(results)
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text('학습 완료 — PNG 저장 가능')

    def _on_error(self, msg):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        QMessageBox.critical(self, '학습 오류', msg)

    def render_results(self, results):
        x_plot = np.linspace(-3, 3, 300).reshape(-1, 1).astype(np.float32)
        adv = results.get('advanced', {})

        # 심화 결과가 있으면 2행 레이아웃, 없으면 1행
        self._figure.clear()
        self._figure.patch.set_facecolor('#12121e')
        if adv:
            all_axes = self._figure.subplots(2, 3)
            main_ax = all_axes[0]
            adv_ax = all_axes[1]
        else:
            main_ax = self._figure.subplots(1, 3)
            adv_ax = []
        for ax in list(main_ax) + list(adv_ax):
            ax.set_facecolor('#12121e')

        # Row 1-1: 기본 함수 근사
        if 'basic' in results:
            r = results['basic']
            fn = r['func']
            y_true = _FUNCTIONS[fn](x_plot)
            y_pred = r['model'].predict(x_plot, verbose=0)
            main_ax[0].plot(x_plot, y_true, 'b-', label='실제값', linewidth=2)
            main_ax[0].plot(x_plot, y_pred, 'r--', label='NN 예측', linewidth=2)
            main_ax[0].set_title(f'기본 근사: {fn}', color='white', fontsize=10)
            main_ax[0].legend(fontsize=8)

        # Row 1-2: 크기 비교
        if 'size_comparison' in results:
            colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
            for (name, r), c in zip(results['size_comparison'].items(), colors):
                main_ax[1].plot(x_plot, r['model'].predict(x_plot, verbose=0),
                                label=name, color=c, linewidth=1.5)
            fn = results.get('basic', {}).get('func', 'sin(x)')
            y_ref = _FUNCTIONS[fn](x_plot)
            main_ax[1].plot(x_plot, y_ref, 'w--', label=f'실제 {fn}',
                            linewidth=1, alpha=0.5)
            main_ax[1].set_title('네트워크 크기 비교', color='white', fontsize=10)
            main_ax[1].legend(fontsize=7)

        # Row 1-3: 극한 복잡도
        if 'extreme' in results:
            r = results['extreme']
            y_pred_ex = r['model'].predict(x_plot, verbose=0)
            y_true_ex = _EXTREME_FN(x_plot)
            main_ax[2].plot(x_plot, y_true_ex, 'b-', label='실제값', linewidth=2)
            main_ax[2].plot(x_plot, y_pred_ex, 'r--', label='NN 예측', linewidth=2)
            main_ax[2].set_title('극한 복잡도 테스트', color='white', fontsize=10)
            main_ax[2].legend(fontsize=8)

        # Row 2: 심화 함수 (tanh, x³)
        for i, (fn_name, r) in enumerate(adv.items()):
            if i >= 3:
                break
            ax = adv_ax[i]
            ax.plot(r['x_plot'], r['y_true'], 'b-', label='실제값', linewidth=2)
            ax.plot(r['x_plot'], r['y_pred'], 'r--', label='NN 예측', linewidth=2)
            ax.set_title(f'심화: {fn_name}  (loss={r["final_loss"]:.4f})',
                         color='#ffd93d', fontsize=10)
            ax.legend(fontsize=8)
        for i in range(len(adv), len(adv_ax)):
            adv_ax[i].axis('off')

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        suffix = '_advanced' if self._results and self._results.get('advanced') else ''
        path = os.path.join(OUTPUTS_DIR, f'01_lab1_results{suffix}.png')
        self._figure.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        return path

    def _save(self):
        try:
            path = self.save_outputs()
            mw = self.window()
            if hasattr(mw, 'set_status_text'):
                mw.set_status_text(f'저장 완료: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, '저장 오류', str(e))
