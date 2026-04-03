import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from tensorflow import keras

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QMessageBox,
)

from core.models import build_lab3_models
from core.trainer import TrainingWorker, make_keras_callback
from labs.lab1_1d import _tooltip_btn

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
_MODEL_NAMES = ['Underfit [4]', 'Good Fit [32,16]', 'Overfit [256,128,64,32]']
_MODEL_COLORS = ['#ff6b6b', '#6bcb77', '#ffd93d']


def generate_overfitting_data(n_samples=200):
    """y = sin(2x) + 0.5x + noise. Shuffled. X: (n,1), y: (n,1)."""
    x = np.linspace(-3, 3, n_samples)
    y = np.sin(2 * x) + 0.5 * x + np.random.normal(0, 0.3, n_samples)
    idx = np.random.permutation(n_samples)
    return x[idx].reshape(-1, 1).astype(np.float32), y[idx].reshape(-1, 1).astype(np.float32)


def lab3_experiment(params, should_stop, emit_progress):
    """
    3개 모델을 순차적으로 학습.
    Returns: {'histories', 'predictions', 'X', 'y', 'names', 'colors', 'table_data'}
    """
    epochs = params['epochs']
    lr = params['learning_rate']
    l1 = params.get('l1_reg', 0.0)
    l2 = params.get('l2_reg', 0.0)

    X, y = generate_overfitting_data()
    underfit, goodfit, overfit = build_lab3_models(l1_reg=l1, l2_reg=l2)

    histories = []
    models_trained = []
    stage_labels = [
        ('1/3 (Underfit)', underfit),
        ('2/3 (Good Fit)', goodfit),
        ('3/3 (Overfit)', overfit),
    ]

    for stage_label, model in stage_labels:
        if should_stop():
            break
        model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
        cb = make_keras_callback(stage_label, epochs, should_stop, emit_progress)
        h = model.fit(X, y, epochs=epochs, validation_split=0.2, callbacks=[cb], verbose=0)
        histories.append(h)
        models_trained.append(model)

    # Pre-compute predictions (avoid main-thread inference)
    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1).astype(np.float32)
    predictions = []
    error_hists = []
    for m in models_trained:
        y_plot = m.predict(x_plot, verbose=0)
        y_train_pred = m.predict(X, verbose=0)
        predictions.append(y_plot)
        error_hists.append(np.abs(y - y_train_pred).flatten())

    names = _MODEL_NAMES[:len(models_trained)]
    colors = _MODEL_COLORS[:len(models_trained)]
    table_data = []
    for h, name in zip(histories, names):
        train_mse = h.history['loss'][-1]
        val_mse = h.history.get('val_loss', [float('nan')])[-1]
        table_data.append([name, f'{train_mse:.4f}', f'{val_mse:.4f}'])

    return {
        'histories': histories,
        'predictions': predictions,
        'error_hists': error_hists,
        'X': X,
        'y': y,
        'x_plot': x_plot,
        'names': names,
        'colors': colors,
        'table_data': table_data,
        'advanced': params.get('advanced', False),
        'l1_reg': l1,
        'l2_reg': l2,
    }


class Lab3Widget(QWidget):
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

        lab_hdr = QLabel('Lab 3: 과적합 vs 과소적합')
        lab_hdr.setStyleSheet('color: #ff7f7f; font-size: 10px;')
        pl.addWidget(lab_hdr)

        for attr, label, default, tip in [
            ('_epochs_input', 'Epochs', '200', '3개 모델 각각에 적용되는 에포크 수.'),
            ('_lr_input', 'Learning Rate', '0.001', 'Adam optimizer learning rate.'),
        ]:
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('color: #aaa; font-size: 10px;')
            r.addWidget(lbl)
            r.addWidget(_tooltip_btn(tip))
            pl.addLayout(r)
            inp = QLineEdit(default)
            inp.setStyleSheet(
                'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
                'border: 1px solid #444; padding: 3px; }'
            )
            pl.addWidget(inp)
            setattr(self, attr, inp)

        # Advanced
        adv_lbl = QLabel('심화 과제')
        adv_lbl.setStyleSheet(
            'color: #7fb8ff; font-size: 10px; font-weight: bold;'
            'border-top: 1px solid #333; padding-top: 8px; margin-top: 4px;'
        )
        pl.addWidget(adv_lbl)
        self._advanced_check = QCheckBox('L1/L2 Regularization')
        self._advanced_check.setStyleSheet('color: #aaa; font-size: 10px;')
        self._advanced_check.toggled.connect(self._toggle_reg)
        pl.addWidget(self._advanced_check)

        self._reg_widget = QWidget()
        rw_l = QVBoxLayout(self._reg_widget)
        rw_l.setContentsMargins(0, 0, 0, 0)
        for attr, label, tip in [
            ('_l1_input', 'L1 λ', 'L1 regularization 계수 (Good Fit 모델에 적용)'),
            ('_l2_input', 'L2 λ', 'L2 regularization 계수 (Good Fit 모델에 적용)'),
        ]:
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('color: #aaa; font-size: 10px;')
            r.addWidget(lbl)
            r.addWidget(_tooltip_btn(tip))
            rw_l.addLayout(r)
            inp = QLineEdit('0.01')
            inp.setStyleSheet(
                'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
                'border: 1px solid #444; padding: 3px; }'
            )
            rw_l.addWidget(inp)
            setattr(self, attr, inp)
        self._reg_widget.setVisible(False)
        pl.addWidget(self._reg_widget)

        pl.addStretch()

        self._start_btn = QPushButton('▶  학습 시작 (3개 순차)')
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
        self._figure, self._axes = plt.subplots(2, 2, figsize=(12, 8))
        self._figure.patch.set_facecolor('#12121e')
        for row in self._axes:
            for ax in row:
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
        )
        self._save_btn.clicked.connect(self._save)
        bottom.addWidget(self._save_btn)
        gl.addLayout(bottom)

        root.addWidget(gp)

    def _toggle_reg(self, checked):
        self._reg_widget.setVisible(checked)

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
        l1, l2 = 0.0, 0.0
        if self._advanced_check.isChecked():
            try:
                l1 = float(self._l1_input.text())
                l2 = float(self._l2_input.text())
            except ValueError:
                raise ValueError('L1/L2 λ는 숫자여야 합니다.')
        return {
            'epochs': epochs, 'learning_rate': lr,
            'l1_reg': l1, 'l2_reg': l2,
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
        self._worker = TrainingWorker(lab3_experiment, params)
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
        try:
            self.render_results(results)
            self._save_btn.setEnabled(True)
        except Exception:
            pass
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text('학습 완료 — PNG 저장 가능')
        self._worker.deleteLater()
        self._worker = None

    def _on_error(self, msg):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        QMessageBox.critical(self, '학습 오류', msg)

    def render_results(self, results):
        if not results.get('histories'):
            return
        histories = results['histories']
        predictions = results['predictions']
        error_hists = results['error_hists']
        X, y = results['X'], results['y']
        x_plot = results['x_plot']
        names = results['names']
        colors = results['colors']
        table_data = results['table_data']

        for row in self._axes:
            for ax in row:
                ax.clear()
                ax.set_facecolor('#12121e')

        # Plot 1: 예측 비교
        self._axes[0, 0].scatter(X, y, color='white', s=10, alpha=0.4, label='데이터')
        for y_p, name, color in zip(predictions, names, colors):
            self._axes[0, 0].plot(x_plot, y_p, color=color, linewidth=2, label=name)
        self._axes[0, 0].set_title('모델 예측 비교', color='white', fontsize=10)
        self._axes[0, 0].legend(fontsize=8)

        # Plot 2: 학습 곡선
        for h, name, color in zip(histories, names, colors):
            self._axes[0, 1].plot(h.history['loss'], color=color, linewidth=1.5,
                                  label=f'{name} train')
            if 'val_loss' in h.history:
                self._axes[0, 1].plot(h.history['val_loss'], color=color,
                                      linewidth=1.5, linestyle='--', alpha=0.6,
                                      label=f'{name} val')
        reg_info = ''
        if results.get('advanced'):
            l1, l2 = results.get('l1_reg', 0.0), results.get('l2_reg', 0.0)
            reg_info = f' [L1={l1}, L2={l2}]'
        self._axes[0, 1].set_title(f'Train vs Val Loss{reg_info}', color='#ffd93d' if reg_info else 'white', fontsize=10)
        self._axes[0, 1].set_yscale('log')
        self._axes[0, 1].legend(fontsize=7)

        # Plot 3: 오차 분포
        for errors, name, color in zip(error_hists, names, colors):
            self._axes[1, 0].hist(errors, bins=20, alpha=0.6, color=color, label=name)
        self._axes[1, 0].set_title('오차 분포 (MAE)', color='white', fontsize=10)
        self._axes[1, 0].legend(fontsize=8)

        # Plot 4: 성능 비교표
        self._axes[1, 1].axis('off')
        col_labels = ['모델', 'Train MSE', 'Val MSE']
        t = self._axes[1, 1].table(
            cellText=table_data, colLabels=col_labels,
            loc='center', cellLoc='center',
        )
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.5)
        for (row, col), cell in t.get_celld().items():
            cell.set_facecolor('#1e1e2e')
            cell.set_edgecolor('#444')
            cell.get_text().set_color('white')
        self._axes[1, 1].set_title('성능 비교', color='white', fontsize=10)

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        suffix = '_advanced' if self._results and self._results.get('advanced') else ''
        path = os.path.join(OUTPUTS_DIR, f'03_overfitting_results{suffix}.png')
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
