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

from core.models import build_lab2_model
from core.trainer import TrainingWorker, make_keras_callback
from labs.lab1_1d import _tooltip_btn   # 재사용

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
G = 9.81

_CONDITIONS = [
    {'v0': 20.0, 'theta_deg': 30.0, 'label': 'v₀=20 θ=30°'},
    {'v0': 30.0, 'theta_deg': 45.0, 'label': 'v₀=30 θ=45° (최대 사거리)'},
    {'v0': 40.0, 'theta_deg': 60.0, 'label': 'v₀=40 θ=60°'},
]


def generate_projectile_data(n_samples=2000, drag=0.0):
    """
    Input X: (n, 3) — [v0 (m/s), theta (deg), t (s)]
    Output y: (n, 2) — [x (m), y (m)]
    drag: air resistance coefficient (0 = no drag)
    """
    v0 = np.random.uniform(10, 50, n_samples)
    theta_deg = np.random.uniform(10, 80, n_samples)
    theta_rad = theta_deg * np.pi / 180
    t_flight = 2 * v0 * np.sin(theta_rad) / G
    t = np.random.uniform(0, 1, n_samples) * t_flight

    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)

    if drag > 0:
        # 단순 지수 감쇠 근사
        x_pos = vx / drag * (1 - np.exp(-drag * t))
        y_pos = ((vy + G / drag) / drag * (1 - np.exp(-drag * t))
                 - G * t / drag)
    else:
        x_pos = vx * t
        y_pos = vy * t - 0.5 * G * t ** 2

    noise = 0.01
    x_pos += np.random.normal(0, noise * (np.abs(x_pos) + 1), n_samples)
    y_pos += np.random.normal(0, noise * (np.abs(y_pos) + 1), n_samples)

    X = np.column_stack([v0, theta_deg, t]).astype(np.float32)
    y = np.column_stack([x_pos, y_pos]).astype(np.float32)
    return X, y


def lab2_experiment(params, should_stop, emit_progress):
    epochs = params['epochs']
    lr = params['learning_rate']
    drag = params.get('drag', 0.0)

    X, y = generate_projectile_data(n_samples=2000, drag=drag)
    model = build_lab2_model()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse',
                  metrics=['mae'])
    cb = make_keras_callback('포물선 학습', epochs, should_stop, emit_progress)
    history = model.fit(X, y, epochs=epochs, validation_split=0.2,
                        batch_size=64, callbacks=[cb], verbose=0)

    # Pre-compute trajectories for rendering (avoid main-thread inference)
    trajectories = []
    for cond in _CONDITIONS:
        v0_c, theta_deg_c = cond['v0'], cond['theta_deg']
        theta_rad_c = theta_deg_c * np.pi / 180
        t_flight_c = 2 * v0_c * np.sin(theta_rad_c) / G
        t_vals = np.linspace(0, t_flight_c, 60).astype(np.float32)
        X_pred = np.column_stack([
            np.full_like(t_vals, v0_c),
            np.full_like(t_vals, theta_deg_c),
            t_vals,
        ])
        xy_pred = model.predict(X_pred, verbose=0)
        x_real = v0_c * np.cos(theta_rad_c) * t_vals
        y_real = v0_c * np.sin(theta_rad_c) * t_vals - 0.5 * G * t_vals ** 2
        trajectories.append({
            'label': cond['label'],
            't_vals': t_vals,
            'x_real': x_real,
            'y_real': y_real,
            'xy_pred': xy_pred,
        })
    return {'model': model, 'history': history, 'drag': drag, 'trajectories': trajectories}


class Lab2Widget(QWidget):
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

        lab_hdr = QLabel('Lab 2: 포물선 운동')
        lab_hdr.setStyleSheet('color: #ffb87f; font-size: 10px;')
        pl.addWidget(lab_hdr)

        for attr, label, default, tip in [
            ('_epochs_input', 'Epochs', '500',
             '학습 에포크 수. 2000 샘플 기준.'),
            ('_lr_input', 'Learning Rate', '0.001',
             'Adam optimizer learning rate.\n권장: 0.0001 ~ 0.01'),
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
        self._advanced_check = QCheckBox('공기 저항 포함')
        self._advanced_check.setStyleSheet('color: #aaa; font-size: 10px;')
        self._advanced_check.toggled.connect(self._toggle_drag)
        pl.addWidget(self._advanced_check)

        self._drag_widget = QWidget()
        dw_l = QVBoxLayout(self._drag_widget)
        dw_l.setContentsMargins(0, 0, 0, 0)
        dr = QHBoxLayout()
        dl = QLabel('Drag Coefficient')
        dl.setStyleSheet('color: #aaa; font-size: 10px;')
        dr.addWidget(dl)
        dr.addWidget(_tooltip_btn('공기 저항 계수 (0.0 ~ 1.0)'))
        dw_l.addLayout(dr)
        self._drag_input = QLineEdit('0.1')
        self._drag_input.setStyleSheet(
            'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
            'border: 1px solid #444; padding: 3px; }'
        )
        dw_l.addWidget(self._drag_input)
        self._drag_widget.setVisible(False)
        pl.addWidget(self._drag_widget)

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
        self._figure, self._axes = plt.subplots(1, 2, figsize=(12, 5))
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
        )
        self._save_btn.clicked.connect(self._save)
        bottom.addWidget(self._save_btn)
        gl.addLayout(bottom)

        root.addWidget(gp)

    def _toggle_drag(self, checked):
        self._drag_widget.setVisible(checked)

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
        drag = 0.0
        if self._advanced_check.isChecked():
            try:
                drag = float(self._drag_input.text())
            except ValueError:
                raise ValueError('Drag Coefficient는 숫자여야 합니다.')
            if drag <= 0:
                raise ValueError('Drag Coefficient는 0보다 커야 합니다. (공기 저항 활성화 시)')
        return {'epochs': epochs, 'learning_rate': lr, 'drag': drag,
                'advanced': self._advanced_check.isChecked()}

    def _start(self):
        try:
            params = self.get_params()
        except ValueError as e:
            QMessageBox.warning(self, '입력 오류', str(e))
            return
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._worker = TrainingWorker(lab2_experiment, params)
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
        if self._worker:
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
        drag = results.get('drag', 0.0)
        history = results['history']

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor('#12121e')

        # Subplot 1: 궤적 비교 (pre-computed in worker)
        colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
        trajectories = results.get('trajectories', [])
        for traj, color in zip(trajectories, colors):
            mask = traj['y_real'] >= 0
            self._axes[0].plot(traj['x_real'][mask], traj['y_real'][mask], '--',
                               color=color, linewidth=1.5, alpha=0.6, label=f'실제 {traj["label"]}')
            mask_p = traj['xy_pred'][:, 1] >= 0
            self._axes[0].plot(traj['xy_pred'][mask_p, 0], traj['xy_pred'][mask_p, 1], '-',
                               color=color, linewidth=2, label=f'NN {traj["label"]}')

        drag_info = f' (공기 저항 drag={drag:.2f})' if drag > 0 else ''
        self._axes[0].set_title(f'포물선 궤적: 실제 vs NN 예측{drag_info}', color='white', fontsize=10)
        self._axes[0].set_xlabel('x (m)', color='white')
        self._axes[0].set_ylabel('y (m)', color='white')
        self._axes[0].legend(fontsize=7)

        # Subplot 2: 학습 곡선
        self._axes[1].plot(history.history['loss'], label='Train Loss', color='#4d96ff')
        if 'val_loss' in history.history:
            self._axes[1].plot(history.history['val_loss'], label='Val Loss', color='#ff6b6b')
        self._axes[1].set_title('학습 곡선', color='white', fontsize=10)
        self._axes[1].set_xlabel('Epoch', color='white')
        self._axes[1].set_ylabel('MSE Loss', color='white')
        self._axes[1].legend(fontsize=8)
        self._axes[1].set_yscale('log')

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        suffix = '_advanced' if self._results and self._results.get('drag', 0) > 0 else ''
        path = os.path.join(OUTPUTS_DIR, f'02_projectile_results{suffix}.png')
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
