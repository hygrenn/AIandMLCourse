import os
import math
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from tensorflow import keras

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QMessageBox, QGroupBox,
)

from core.models import build_lab4_model
from core.trainer import TrainingWorker, make_keras_callback
from labs.lab1_1d import _tooltip_btn

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
G = 9.81
_DEFAULT_LENGTHS = [0.5, 1.0, 2.0]


def rk4_pendulum(L, theta0_rad, gamma=0.0, t_max=10.0, dt=0.01):
    """
    RK4 integration: d²θ/dt² = -(g/L)sin(θ) - γ·(dθ/dt)
    Returns: (t_array, theta_array, omega_array)
    """
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    theta = np.zeros(n)
    omega = np.zeros(n)
    theta[0] = theta0_rad

    def deriv(th, om):
        return om, -(G / L) * math.sin(th) - gamma * om

    for i in range(n - 1):
        k1_th, k1_om = deriv(theta[i], omega[i])
        k2_th, k2_om = deriv(theta[i] + dt / 2 * k1_th, omega[i] + dt / 2 * k1_om)
        k3_th, k3_om = deriv(theta[i] + dt / 2 * k2_th, omega[i] + dt / 2 * k2_om)
        k4_th, k4_om = deriv(theta[i] + dt * k3_th, omega[i] + dt * k3_om)
        theta[i + 1] = theta[i] + dt / 6 * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = omega[i] + dt / 6 * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return t, theta, omega


def _analytical_period(L, theta0_rad, gamma=0.0):
    """
    T ≈ T0 * (1 + k²/4 + 9k⁴/64) for undamped pendulum (Bessel approximation).
    For damped case: T_damped = T_undamped / sqrt(1 - (γ/(2ω₀))²)
    where ω₀ = sqrt(g/L). Returns inf if overdamped (γ >= 2ω₀).
    """
    T0 = 2 * math.pi * math.sqrt(L / G)
    k = math.sin(theta0_rad / 2)
    T_undamped = T0 * (1 + k ** 2 / 4 + 9 * k ** 4 / 64)
    if gamma == 0.0:
        return T_undamped
    omega0 = math.sqrt(G / L)
    discriminant = 1.0 - (gamma / (2 * omega0)) ** 2
    if discriminant <= 0:
        return float('inf')  # overdamped: no oscillation
    return T_undamped / math.sqrt(discriminant)


def generate_pendulum_data(n_samples=2000, gamma=0.0):
    """
    X: (n, 2) — [L (m), theta0 (deg)]
    y: (n, 1) — [T (s)]
    """
    L = np.random.uniform(0.1, 3.0, n_samples)
    theta0_deg = np.random.uniform(5, 80, n_samples)
    theta0_rad = theta0_deg * math.pi / 180

    T = np.array([_analytical_period(l, th, gamma) for l, th in zip(L, theta0_rad)])
    T += np.random.normal(0, 0.002 * T, n_samples)

    X = np.column_stack([L, theta0_deg]).astype(np.float32)
    y = T.reshape(-1, 1).astype(np.float32)
    return X, y


def lab4_experiment(params, should_stop, emit_progress):
    epochs = params['epochs']
    lr = params['learning_rate']
    gamma = params.get('gamma', 0.0)
    lengths = params.get('lengths', _DEFAULT_LENGTHS)

    X, y = generate_pendulum_data(n_samples=2000, gamma=gamma)
    model = build_lab4_model()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    cb = make_keras_callback('진자 주기 학습', epochs, should_stop, emit_progress)
    history = model.fit(X, y, epochs=epochs, validation_split=0.2,
                        batch_size=64, callbacks=[cb], verbose=0)

    if should_stop():
        return {'history': None}

    # RK4 simulation (L=1.0, theta=20° as reference)
    t_rk4, theta_rk4, omega_rk4 = rk4_pendulum(
        1.0, 20 * math.pi / 180, gamma=gamma, t_max=10.0, dt=0.01
    )

    # Pre-compute period predictions for each selected length
    theta_range = np.linspace(5, 80, 60)
    period_curves = []
    for L in lengths:
        T_analytical = np.array([
            _analytical_period(L, th * math.pi / 180, gamma)
            for th in theta_range
        ])
        X_pred = np.column_stack([
            np.full(len(theta_range), L),
            theta_range,
        ]).astype(np.float32)
        T_pred = model.predict(X_pred, verbose=0).flatten()
        period_curves.append({
            'L': L,
            'theta_range': theta_range,
            'T_analytical': T_analytical,
            'T_pred': T_pred,
        })

    return {
        'history': history,
        'rk4': {
            't': t_rk4,
            'theta': theta_rk4,
            'omega': omega_rk4,
            'L': 1.0,
            'gamma': gamma,
        },
        'period_curves': period_curves,
        'gamma': gamma,
    }


class Lab4Widget(QWidget):
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

        lab_hdr = QLabel('Lab 4: 진자 주기 예측')
        lab_hdr.setStyleSheet('color: #b87fff; font-size: 10px;')
        pl.addWidget(lab_hdr)

        # Length checkboxes
        len_grp = QGroupBox('진자 길이 선택')
        len_grp.setStyleSheet(
            'QGroupBox { color: #aaa; font-size: 10px; border: 1px solid #333; padding: 4px; }'
        )
        lg = QVBoxLayout(len_grp)
        lg.setContentsMargins(4, 4, 4, 4)
        self._len_checks = {}
        for L in _DEFAULT_LENGTHS:
            cb = QCheckBox(f'L = {L} m')
            cb.setChecked(True)
            cb.setStyleSheet('color: #aaa; font-size: 10px;')
            lg.addWidget(cb)
            self._len_checks[L] = cb
        pl.addWidget(len_grp)

        for attr, label, default, tip in [
            ('_epochs_input', 'Epochs', '500', '학습 에포크 수.'),
            ('_lr_input', 'Learning Rate', '0.001', 'Adam learning rate.'),
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
        self._advanced_check = QCheckBox('감쇠 진자 (Damped Pendulum)')
        self._advanced_check.setStyleSheet('color: #aaa; font-size: 10px;')
        self._advanced_check.toggled.connect(self._toggle_damping)
        pl.addWidget(self._advanced_check)

        self._damp_widget = QWidget()
        dw_l = QVBoxLayout(self._damp_widget)
        dw_l.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        l = QLabel('감쇠 계수 γ')
        l.setStyleSheet('color: #aaa; font-size: 10px;')
        r.addWidget(l)
        r.addWidget(_tooltip_btn('감쇠 계수 (0.0 = 비감쇠). d²θ/dt² = -(g/L)sin(θ) - γ·dθ/dt'))
        dw_l.addLayout(r)
        self._gamma_input = QLineEdit('0.1')
        self._gamma_input.setStyleSheet(
            'QLineEdit { background: #2a2a3e; color: #ccc; font-size: 10px;'
            'border: 1px solid #444; padding: 3px; }'
        )
        dw_l.addWidget(self._gamma_input)
        self._damp_widget.setVisible(False)
        pl.addWidget(self._damp_widget)

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
        self._figure, self._axes = plt.subplots(1, 3, figsize=(14, 5))
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

    def _toggle_damping(self, checked):
        self._damp_widget.setVisible(checked)

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
        lengths = [L for L, cb in self._len_checks.items() if cb.isChecked()]
        if not lengths:
            raise ValueError('진자 길이를 하나 이상 선택해야 합니다.')
        gamma = 0.0
        if self._advanced_check.isChecked():
            try:
                gamma = float(self._gamma_input.text())
            except ValueError:
                raise ValueError('감쇠 계수 γ는 숫자여야 합니다.')
            if gamma < 0:
                raise ValueError('감쇠 계수 γ는 0 이상이어야 합니다.')
        return {
            'epochs': epochs, 'learning_rate': lr,
            'lengths': lengths, 'gamma': gamma,
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
        self._worker = TrainingWorker(lab4_experiment, params)
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
        except Exception as e:
            QMessageBox.warning(self, '렌더링 오류', str(e))
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text('학습 완료 — PNG 저장 가능')
        self._worker.deleteLater()
        self._worker = None

    def _on_error(self, msg):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._worker.deleteLater()
        self._worker = None
        QMessageBox.critical(self, '학습 오류', msg)

    def render_results(self, results):
        if not results.get('history'):
            return

        history = results['history']
        rk4 = results['rk4']
        period_curves = results.get('period_curves', [])
        gamma = results.get('gamma', 0.0)

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor('#12121e')

        colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']

        # Subplot 1: 주기 예측 (NN vs 해석해) — pre-computed
        for curve, color in zip(period_curves, colors):
            self._axes[0].plot(curve['theta_range'], curve['T_analytical'], '--',
                               color=color, linewidth=1.5, alpha=0.6,
                               label=f'L={curve["L"]}m 해석해')
            self._axes[0].plot(curve['theta_range'], curve['T_pred'], '-',
                               color=color, linewidth=2, label=f'L={curve["L"]}m NN')
        period_title = f'주기 예측: NN vs 해석해 (감쇠 γ={gamma:.2f})' if gamma > 0 else '주기 예측: NN vs 해석해'
        self._axes[0].set_title(period_title, color='#ffd93d' if gamma > 0 else 'white', fontsize=10)
        self._axes[0].set_xlabel('각도 (°)', color='white')
        self._axes[0].set_ylabel('주기 T (s)', color='white')
        self._axes[0].legend(fontsize=7)

        # Subplot 2: RK4 시뮬레이션
        t, theta_rk, omega_rk = rk4['t'], rk4['theta'], rk4['omega']
        theta_deg = np.degrees(theta_rk)
        self._axes[1].plot(t, theta_deg, color='#4d96ff', linewidth=1.5)
        self._axes[1].set_title(
            f'RK4 시뮬레이션 (L={rk4["L"]}m, γ={gamma:.2f})', color='white', fontsize=10
        )
        self._axes[1].set_xlabel('시간 (s)', color='white')
        self._axes[1].set_ylabel('각도 (°)', color='white')

        # Phase space inset
        ax_inset = self._axes[1].inset_axes([0.55, 0.55, 0.42, 0.42])
        ax_inset.plot(theta_deg, np.degrees(omega_rk), color='#ff6b6b', linewidth=1)
        ax_inset.set_facecolor('#0a0a1e')
        ax_inset.set_xlabel('θ (°)', color='white', fontsize=7)
        ax_inset.set_ylabel('ω (°/s)', color='white', fontsize=7)
        ax_inset.tick_params(colors='white', labelsize=6)

        # Subplot 3: 학습 곡선
        self._axes[2].plot(history.history['loss'], color='#4d96ff', label='Train Loss')
        if 'val_loss' in history.history:
            self._axes[2].plot(history.history['val_loss'], color='#ff6b6b',
                               linestyle='--', label='Val Loss')
        self._axes[2].set_title('학습 곡선', color='white', fontsize=10)
        self._axes[2].set_xlabel('Epoch', color='white')
        self._axes[2].set_ylabel('MSE Loss', color='white')
        self._axes[2].set_yscale('log')
        self._axes[2].legend(fontsize=8)

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        gamma = self._results.get('gamma', 0.0) if self._results else 0.0
        suffix = '_advanced' if gamma > 0 else ''
        path = os.path.join(OUTPUTS_DIR, f'04_pendulum_results{suffix}.png')
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
