# Physics NN Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PySide6 GUI 앱으로 4개 물리 Neural Network Lab(1D 함수 근사, 포물선 운동, 과적합 분석, 진자 주기)을 사이드바 네비게이션으로 통합 구현

**Architecture:** Lab별 독립 QWidget + 공통 QThread 기반 TrainingWorker. 각 Lab은 `experiment_fn(params, should_stop, emit_progress) → dict`를 정의하고 TrainingWorker가 백그라운드 실행. matplotlib FigureCanvasQTAgg로 그래프 임베드, 다크 테마.

**Tech Stack:** PySide6 6.x, TensorFlow/Keras, matplotlib (FigureCanvasQTAgg + NavigationToolbar2QT), numpy, scipy, pytest, pytest-qt

---

## File Map

| 파일 | 역할 |
|------|------|
| `main.py` | QApplication, MainWindow, 사이드바, 다크 테마, 상태바 |
| `core/__init__.py` | 빈 파일 |
| `core/trainer.py` | `TrainingWorker(QThread)`, `make_keras_callback` |
| `core/models.py` | Keras 모델 팩토리 함수 (4개 Lab) |
| `labs/__init__.py` | 빈 파일 |
| `labs/lab1_1d.py` | `Lab1Widget`, `generate_lab1_data`, `lab1_experiment` |
| `labs/lab2_projectile.py` | `Lab2Widget`, `generate_projectile_data`, `lab2_experiment` |
| `labs/lab3_overfitting.py` | `Lab3Widget`, `generate_overfitting_data`, `lab3_experiment` |
| `labs/lab4_pendulum.py` | `Lab4Widget`, `generate_pendulum_data`, `rk4_pendulum`, `lab4_experiment` |
| `tests/conftest.py` | pytest QApplication fixture |
| `tests/test_models.py` | 모델 팩토리 단위 테스트 |
| `tests/test_data.py` | 데이터 생성 단위 테스트 |
| `tests/test_trainer.py` | TrainingWorker 단위 테스트 |
| `outputs/` | PNG 자동 저장 디렉토리 |
| `pyproject.toml` | 의존성 선언 |

---

### Task 1: 프로젝트 구조 셋업

**Files:**
- Create: `pyproject.toml`
- Create: `core/__init__.py`
- Create: `labs/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `outputs/.gitkeep`

- [ ] **Step 1: pyproject.toml 생성**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "physics-nn-explorer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pyside6",
    "tensorflow",
    "numpy",
    "matplotlib",
    "scipy",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-qt"]

[tool.hatch.build.targets.wheel]
packages = ["core", "labs"]
```

- [ ] **Step 2: 디렉토리 및 빈 파일 생성**

```bash
mkdir -p core labs tests outputs
touch core/__init__.py labs/__init__.py tests/__init__.py outputs/.gitkeep
```

- [ ] **Step 3: tests/conftest.py 생성**

```python
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope='session')
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance
```

- [ ] **Step 4: 설치 및 임포트 확인**

```bash
uv sync --extra dev
python -c "import PySide6; import tensorflow; import matplotlib; import scipy; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml core/ labs/ tests/ outputs/
git commit -m "feat: project scaffold — directories and dependencies"
```

---

### Task 2: core/models.py — Keras 모델 팩토리

**Files:**
- Create: `core/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: tests/test_models.py 작성**

```python
import numpy as np
import pytest


def test_lab1_model_output_shape():
    from core.models import build_lab1_model
    model = build_lab1_model()
    model.compile(optimizer='adam', loss='mse')
    result = model.predict(np.array([[0.5]]), verbose=0)
    assert result.shape == (1, 1)


def test_lab1_size_models_keys():
    from core.models import build_lab1_size_models
    models = build_lab1_size_models()
    assert set(models.keys()) == {
        'Small [32]', 'Medium [64,64]', 'Large [128,128]', 'Very Large [128,128,64]'
    }


def test_lab1_extreme_model_output_shape():
    from core.models import build_lab1_extreme_model
    model = build_lab1_extreme_model()
    model.compile(optimizer='adam', loss='mse')
    result = model.predict(np.array([[1.0]]), verbose=0)
    assert result.shape == (1, 1)


def test_lab2_model_output_shape():
    from core.models import build_lab2_model
    model = build_lab2_model()
    model.compile(optimizer='adam', loss='mse')
    result = model.predict(np.array([[20.0, 45.0, 1.0]]), verbose=0)
    assert result.shape == (1, 2)


def test_lab3_models_shapes():
    from core.models import build_lab3_models
    underfit, goodfit, overfit = build_lab3_models()
    for m in (underfit, goodfit, overfit):
        m.compile(optimizer='adam', loss='mse')
        result = m.predict(np.array([[0.5]]), verbose=0)
        assert result.shape == (1, 1)


def test_lab3_models_with_regularization():
    from core.models import build_lab3_models
    _, goodfit, _ = build_lab3_models(l1_reg=0.01, l2_reg=0.01)
    goodfit.compile(optimizer='adam', loss='mse')
    result = goodfit.predict(np.array([[0.5]]), verbose=0)
    assert result.shape == (1, 1)


def test_lab4_model_output_shape():
    from core.models import build_lab4_model
    model = build_lab4_model()
    model.compile(optimizer='adam', loss='mse')
    result = model.predict(np.array([[1.0, 30.0]]), verbose=0)
    assert result.shape == (1, 1)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'core.models'`

- [ ] **Step 3: core/models.py 구현**

```python
from tensorflow import keras


def build_lab1_model():
    """[128, 128, 64] tanh. Input: (1,) → Output: (1,)."""
    return keras.Sequential([
        keras.layers.Dense(128, activation='tanh', input_shape=(1,)),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(1, activation='linear'),
    ], name='lab1_model')


def build_lab1_size_models():
    """4가지 크기 모델 dict. Keys: 'Small [32]', 'Medium [64,64]', 'Large [128,128]', 'Very Large [128,128,64]'."""
    configs = {
        'Small [32]': [32],
        'Medium [64,64]': [64, 64],
        'Large [128,128]': [128, 128],
        'Very Large [128,128,64]': [128, 128, 64],
    }
    models = {}
    for name, sizes in configs.items():
        layers = [keras.layers.Dense(sizes[0], activation='tanh', input_shape=(1,))]
        for s in sizes[1:]:
            layers.append(keras.layers.Dense(s, activation='tanh'))
        layers.append(keras.layers.Dense(1, activation='linear'))
        models[name] = keras.Sequential(layers)
    return models


def build_lab1_extreme_model():
    """[256, 256, 128, 64] tanh. Input: (1,) → Output: (1,)."""
    return keras.Sequential([
        keras.layers.Dense(256, activation='tanh', input_shape=(1,)),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(1, activation='linear'),
    ], name='lab1_extreme')


def build_lab2_model():
    """[128, 64, 32] relu+dropout. Input: (3,) [v0, theta_deg, t] → Output: (2,) [x, y]."""
    return keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(3,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation='linear'),
    ], name='lab2_model')


def build_lab3_models(l1_reg=0.0, l2_reg=0.0):
    """
    3개 모델 반환: (underfit, goodfit, overfit).
    Input: (1,) → Output: (1,).
    l1_reg, l2_reg: goodfit 모델에만 regularization 적용.
    """
    reg = keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg) if (l1_reg or l2_reg) else None

    underfit = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(1,)),
        keras.layers.Dense(1, activation='linear'),
    ], name='underfit')

    goodfit = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(1,), kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(1, activation='linear'),
    ], name='goodfit')

    overfit = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(1,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ], name='overfit')

    return underfit, goodfit, overfit


def build_lab4_model():
    """[64, 32, 16] relu+dropout. Input: (2,) [L_m, theta0_deg] → Output: (1,) [T_sec]."""
    return keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear'),
    ], name='lab4_model')
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_models.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/models.py tests/test_models.py
git commit -m "feat: add Keras model factory functions for all 4 labs"
```

---

### Task 3: core/trainer.py — TrainingWorker

**Files:**
- Create: `core/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: tests/test_trainer.py 작성**

```python
import pytest
from unittest.mock import MagicMock


def test_worker_emits_finished(app):
    from core.trainer import TrainingWorker

    def mock_experiment(params, should_stop, emit_progress):
        emit_progress('테스트', 1, 10, 0.5)
        return {'result': 42}

    worker = TrainingWorker(mock_experiment, {})
    received = []
    worker.finished.connect(lambda r: received.append(r))
    worker.start()
    worker.wait(5000)
    assert received == [{'result': 42}]


def test_worker_stop_flag_starts_false(app):
    from core.trainer import TrainingWorker

    stop_values = []

    def mock_experiment(params, should_stop, emit_progress):
        stop_values.append(should_stop())
        return {}

    worker = TrainingWorker(mock_experiment, {})
    worker.start()
    worker.wait(5000)
    assert stop_values[0] is False


def test_worker_emits_error_on_exception(app):
    from core.trainer import TrainingWorker

    def bad_experiment(params, should_stop, emit_progress):
        raise RuntimeError('test error message')

    worker = TrainingWorker(bad_experiment, {})
    errors = []
    worker.error.connect(lambda msg: errors.append(msg))
    worker.start()
    worker.wait(5000)
    assert len(errors) == 1
    assert 'test error message' in errors[0]


def test_keras_callback_stops_training_on_flag():
    from core.trainer import make_keras_callback

    mock_model = MagicMock()
    mock_model.stop_training = False

    callback = make_keras_callback('테스트', 10, lambda: True, lambda *a: None)
    callback.model = mock_model
    callback.on_epoch_end(0, {'loss': 0.1})
    assert mock_model.stop_training is True


def test_keras_callback_does_not_stop_when_flag_false():
    from core.trainer import make_keras_callback

    mock_model = MagicMock()
    mock_model.stop_training = False

    callback = make_keras_callback('테스트', 10, lambda: False, lambda *a: None)
    callback.model = mock_model
    callback.on_epoch_end(0, {'loss': 0.1})
    assert mock_model.stop_training is False


def test_keras_callback_raises_on_nan_loss():
    from core.trainer import make_keras_callback
    import pytest

    mock_model = MagicMock()
    callback = make_keras_callback('테스트', 10, lambda: False, lambda *a: None)
    callback.model = mock_model

    with pytest.raises(ValueError, match='NaN'):
        callback.on_epoch_end(0, {'loss': float('nan')})
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_trainer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'core.trainer'`

- [ ] **Step 3: core/trainer.py 구현**

```python
import math
from PySide6.QtCore import QThread, Signal
from tensorflow import keras


class TrainingWorker(QThread):
    """
    experiment_fn(params, should_stop, emit_progress) → dict 을 백그라운드에서 실행.
      - should_stop: callable() → bool  (stop() 호출 시 True)
      - emit_progress: callable(stage: str, epoch: int, total: int, loss: float)
    """
    progress = Signal(str, int, int, float)   # stage, epoch, total, loss
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, experiment_fn, params):
        super().__init__()
        self._experiment_fn = experiment_fn
        self._params = params
        self._stop = False

    def run(self):
        try:
            results = self._experiment_fn(
                self._params,
                lambda: self._stop,
                lambda stage, ep, total, loss: self.progress.emit(stage, ep, total, loss),
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._stop = True


def make_keras_callback(stage_label, total_epochs, should_stop, emit_progress):
    """
    매 epoch 종료 시:
    - emit_progress(stage_label, epoch+1, total_epochs, loss) 호출
    - should_stop() True이면 model.stop_training = True 설정
    - loss가 NaN이면 ValueError 발생
    """
    class _ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = (logs or {}).get('loss', float('nan'))
            if math.isnan(loss):
                raise ValueError(
                    f'Loss가 NaN이 되었습니다 (epoch {epoch + 1}). Learning Rate를 낮춰보세요.'
                )
            emit_progress(stage_label, epoch + 1, total_epochs, loss)
            if should_stop():
                self.model.stop_training = True

    return _ProgressCallback()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_trainer.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: add TrainingWorker(QThread) and Keras progress callback"
```

---

### Task 4: main.py — MainWindow + 다크 테마 + 사이드바

**Files:**
- Create: `main.py`

- [ ] **Step 1: main.py 작성**

```python
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QStatusBar, QProgressBar,
)
from PySide6.QtGui import QPalette, QColor

PALETTE = {
    'window':       '#1e1e2e',
    'window_text':  '#cdd6f4',
    'base':         '#181825',
    'alt_base':     '#313244',
    'text':         '#cdd6f4',
    'button':       '#313244',
    'button_text':  '#cdd6f4',
    'highlight':    '#7fb8ff',
    'hl_text':      '#1e1e2e',
    'sidebar':      '#1a1a2e',
}


def apply_dark_theme(app):
    app.setStyle('Fusion')
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(PALETTE['window']))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(PALETTE['window_text']))
    p.setColor(QPalette.ColorRole.Base,            QColor(PALETTE['base']))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(PALETTE['alt_base']))
    p.setColor(QPalette.ColorRole.Text,            QColor(PALETTE['text']))
    p.setColor(QPalette.ColorRole.Button,          QColor(PALETTE['button']))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(PALETTE['button_text']))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(PALETTE['highlight']))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(PALETTE['hl_text']))
    app.setPalette(p)


class _SidebarButton(QPushButton):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(58)
        self._title = title
        self._subtitle = subtitle
        self._refresh(False)

    def setChecked(self, checked):
        super().setChecked(checked)
        self._refresh(checked)

    def _refresh(self, active):
        border = '3px solid #7fb8ff' if active else '3px solid transparent'
        bg = '#2d2d5e' if active else 'transparent'
        color = '#ffffff' if active else '#888888'
        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-left: {border};
                background: {bg};
                color: {color};
                text-align: left;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: #252545; color: #cccccc; }}
        """)
        self.setText(f"{self._title}\n{self._subtitle}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Physics NN Explorer')
        self.setMinimumSize(1100, 700)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(150)
        sidebar.setStyleSheet(f'background: {PALETTE["sidebar"]};')
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        hdr = QLabel('🔭  Labs')
        hdr.setStyleSheet(
            'color: #7fb8ff; font-weight: bold; font-size: 13px;'
            'padding: 12px; border-bottom: 1px solid #333;'
        )
        sl.addWidget(hdr)

        self._btns = []
        for i, (icon, title, sub) in enumerate([
            ('🔬', 'Lab 1', '1D 함수 근사'),
            ('🚀', 'Lab 2', '포물선 운동'),
            ('⚖️', 'Lab 3', '과적합 분석'),
            ('🕰️', 'Lab 4', '진자 주기'),
        ]):
            btn = _SidebarButton(f'{icon} {title}', sub)
            btn.clicked.connect(lambda _, idx=i: self._switch(idx))
            sl.addWidget(btn)
            self._btns.append(btn)

        sl.addStretch()
        root.addWidget(sidebar)

        # ── Stack ─────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f'background: {PALETTE["window"]};')
        root.addWidget(self._stack)

        # ── Status bar ────────────────────────────────────────
        sb = QStatusBar()
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            'QProgressBar { background: #222; border: none; }'
            'QProgressBar::chunk { background: #4a7a4a; }'
        )
        self._status_lbl = QLabel('준비')
        self._status_lbl.setStyleSheet('color: #888; font-size: 11px; padding-right: 8px;')
        sb.addWidget(self._progress, 1)
        sb.addPermanentWidget(self._status_lbl)
        self.setStatusBar(sb)

        self._switch(0)

    def _switch(self, index):
        for i, btn in enumerate(self._btns):
            btn.setChecked(i == index)
        if self._stack.count() > index:
            self._stack.setCurrentIndex(index)

    def register_lab(self, widget):
        """LabWidget을 스택에 추가. Task 9에서 호출."""
        self._stack.addWidget(widget)

    def update_status(self, stage, epoch, total, loss):
        """TrainingWorker.progress signal 에 연결."""
        pct = int(epoch / total * 100) if total else 0
        self._progress.setValue(pct)
        self._status_lbl.setText(f'{stage} — Epoch {epoch}/{total} — loss: {loss:.4f}')

    def set_status_text(self, text):
        self._progress.setValue(0)
        self._status_lbl.setText(text)


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 앱 실행 확인**

```bash
python main.py
```

Expected: 사이드바에 4개 버튼이 있는 다크 테마 창이 열림. Lab 버튼 클릭 시 파란 하이라이트 전환.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add MainWindow with dark theme, sidebar, status bar"
```

---

### Task 5: labs/lab1_1d.py — 1D 함수 근사

**Files:**
- Create: `labs/lab1_1d.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: tests/test_data.py에 Lab 1 테스트 작성**

```python
import numpy as np
import pytest


# ── Lab 1 ────────────────────────────────────────────────────────────────────

def test_generate_lab1_data_shape():
    from labs.lab1_1d import generate_lab1_data
    X, y = generate_lab1_data('sin(x)', n_points=100)
    assert X.shape == (100, 1)
    assert y.shape == (100, 1)


def test_generate_lab1_data_values():
    from labs.lab1_1d import generate_lab1_data
    X, y = generate_lab1_data('sin(x)', n_points=50)
    np.testing.assert_allclose(y, np.sin(X), rtol=1e-5)


def test_generate_lab1_data_shuffled():
    from labs.lab1_1d import generate_lab1_data
    X, _ = generate_lab1_data('sin(x)', n_points=100)
    diffs = np.diff(X.flatten())
    assert not np.all(diffs > 0), 'Data should be shuffled, not sorted'


def test_generate_lab1_data_unknown_function():
    from labs.lab1_1d import generate_lab1_data
    with pytest.raises(ValueError, match='Unknown function'):
        generate_lab1_data('mystery_fn')


def test_lab1_get_params_defaults(app):
    from labs.lab1_1d import Lab1Widget
    w = Lab1Widget()
    p = w.get_params()
    assert p['epochs'] == 3000
    assert p['learning_rate'] == pytest.approx(0.001)
    assert p['function'] == 'sin(x)'
    assert p['advanced'] is False


def test_lab1_get_params_invalid_epochs(app):
    from labs.lab1_1d import Lab1Widget
    w = Lab1Widget()
    w._epochs_input.setText('0')
    with pytest.raises(ValueError, match='Epochs'):
        w.get_params()


def test_lab1_get_params_invalid_lr(app):
    from labs.lab1_1d import Lab1Widget
    w = Lab1Widget()
    w._lr_input.setText('-0.001')
    with pytest.raises(ValueError, match='Learning Rate'):
        w.get_params()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_data.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'labs.lab1_1d'`

- [ ] **Step 3: labs/lab1_1d.py 구현**

```python
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
        adv = {}
        for fn_name in ['tanh(x)', 'x³']:
            X_a, y_a = generate_lab1_data(fn_name)
            am = build_lab1_model()
            am.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
            cb = make_keras_callback(f'심화 ({fn_name})', epochs, should_stop, emit_progress)
            ah = am.fit(X_a, y_a, epochs=epochs, validation_split=0.2, callbacks=[cb], verbose=0)
            adv[fn_name] = {'model': am, 'history': ah, 'X': X_a, 'y': y_a}
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
        for ax in self._axes:
            ax.clear()
            ax.set_facecolor('#12121e')

        # Subplot 1: 기본 함수 근사
        if 'basic' in results:
            r = results['basic']
            fn = r['func']
            y_true = _FUNCTIONS[fn](x_plot)
            y_pred = r['model'].predict(x_plot, verbose=0)
            self._axes[0].plot(x_plot, y_true, 'b-', label='실제값', linewidth=2)
            self._axes[0].plot(x_plot, y_pred, 'r--', label='NN 예측', linewidth=2)
            self._axes[0].set_title(f'기본 근사: {fn}', color='white', fontsize=10)
            self._axes[0].legend(fontsize=8)

        # Subplot 2: 크기 비교
        if 'size_comparison' in results:
            colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
            for (name, r), c in zip(results['size_comparison'].items(), colors):
                self._axes[1].plot(x_plot, r['model'].predict(x_plot, verbose=0),
                                   label=name, color=c, linewidth=1.5)
            self._axes[1].plot(x_plot, np.sin(x_plot), 'w--', label='실제 sin(x)',
                               linewidth=1, alpha=0.5)
            self._axes[1].set_title('네트워크 크기 비교', color='white', fontsize=10)
            self._axes[1].legend(fontsize=7)

        # Subplot 3: 극한 복잡도
        if 'extreme' in results:
            r = results['extreme']
            y_pred_ex = r['model'].predict(x_plot, verbose=0)
            y_true_ex = _EXTREME_FN(x_plot)
            self._axes[2].plot(x_plot, y_true_ex, 'b-', label='실제값', linewidth=2)
            self._axes[2].plot(x_plot, y_pred_ex, 'r--', label='NN 예측', linewidth=2)
            self._axes[2].set_title('극한 복잡도 테스트', color='white', fontsize=10)
            self._axes[2].legend(fontsize=8)

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        path = os.path.join(OUTPUTS_DIR, '01_lab1_results.png')
        self._figure.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        return path

    def _save(self):
        path = self.save_outputs()
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text(f'저장 완료: {os.path.basename(path)}')
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add labs/lab1_1d.py tests/test_data.py
git commit -m "feat: add Lab1Widget — 1D function approximation"
```

---

### Task 6: labs/lab2_projectile.py — 포물선 운동

**Files:**
- Create: `labs/lab2_projectile.py`
- Modify: `tests/test_data.py` (Lab 2 테스트 추가)

- [ ] **Step 1: tests/test_data.py에 Lab 2 테스트 추가**

```python
# 파일 끝에 추가

# ── Lab 2 ────────────────────────────────────────────────────────────────────

def test_generate_projectile_data_shape():
    from labs.lab2_projectile import generate_projectile_data
    X, y = generate_projectile_data(n_samples=100)
    assert X.shape == (100, 3)
    assert y.shape == (100, 2)


def test_generate_projectile_data_physics_no_drag():
    """y 좌표는 포물선 운동이므로 음수가 될 수 있음 — 단, 적어도 일부는 양수여야 함."""
    from labs.lab2_projectile import generate_projectile_data
    _, y = generate_projectile_data(n_samples=200)
    assert np.any(y[:, 1] > 0), 'Some y-positions should be positive'


def test_generate_projectile_data_with_drag():
    from labs.lab2_projectile import generate_projectile_data
    X_nd, y_nd = generate_projectile_data(n_samples=100, drag=0.0)
    X_d, y_d = generate_projectile_data(n_samples=100, drag=0.3)
    assert X_d.shape == (100, 3)
    assert y_d.shape == (100, 2)


def test_lab2_get_params_defaults(app):
    from labs.lab2_projectile import Lab2Widget
    w = Lab2Widget()
    p = w.get_params()
    assert p['epochs'] == 500
    assert p['learning_rate'] == pytest.approx(0.001)
    assert p['drag'] == pytest.approx(0.0)
    assert p['advanced'] is False
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_data.py::test_generate_projectile_data_shape -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'labs.lab2_projectile'`

- [ ] **Step 3: labs/lab2_projectile.py 구현**

```python
import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from tensorflow import keras

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QMessageBox, QToolButton,
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

    return {'model': model, 'history': history, 'drag': drag}


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
            if drag < 0:
                raise ValueError('Drag Coefficient는 0 이상이어야 합니다.')
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

    def _on_error(self, msg):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        QMessageBox.critical(self, '학습 오류', msg)

    def render_results(self, results):
        model = results['model']
        drag = results.get('drag', 0.0)
        history = results['history']

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor('#12121e')

        # Subplot 1: 궤적 비교
        colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
        for cond, color in zip(_CONDITIONS, colors):
            v0, theta_deg = cond['v0'], cond['theta_deg']
            theta_rad = theta_deg * np.pi / 180
            t_flight = 2 * v0 * np.sin(theta_rad) / G
            t_vals = np.linspace(0, t_flight, 60).astype(np.float32)

            # 실제 물리
            x_real = v0 * np.cos(theta_rad) * t_vals
            y_real = v0 * np.sin(theta_rad) * t_vals - 0.5 * G * t_vals ** 2
            mask = y_real >= 0
            self._axes[0].plot(x_real[mask], y_real[mask], '--',
                               color=color, linewidth=1.5, alpha=0.6, label=f'실제 {cond["label"]}')

            # NN 예측
            X_pred = np.column_stack([
                np.full_like(t_vals, v0),
                np.full_like(t_vals, theta_deg),
                t_vals,
            ])
            xy_pred = model.predict(X_pred, verbose=0)
            mask_p = xy_pred[:, 1] >= 0
            self._axes[0].plot(xy_pred[mask_p, 0], xy_pred[mask_p, 1], '-',
                               color=color, linewidth=2, label=f'NN {cond["label"]}')

        self._axes[0].set_title('포물선 궤적: 실제 vs NN 예측', color='white', fontsize=10)
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
        path = os.path.join(OUTPUTS_DIR, '02_projectile_results.png')
        self._figure.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        return path

    def _save(self):
        path = self.save_outputs()
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text(f'저장 완료: {os.path.basename(path)}')
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add labs/lab2_projectile.py tests/test_data.py
git commit -m "feat: add Lab2Widget — projectile motion regression"
```

---

### Task 7: labs/lab3_overfitting.py — 과적합 vs 과소적합

**Files:**
- Create: `labs/lab3_overfitting.py`
- Modify: `tests/test_data.py` (Lab 3 테스트 추가)

- [ ] **Step 1: tests/test_data.py에 Lab 3 테스트 추가**

```python
# 파일 끝에 추가

# ── Lab 3 ────────────────────────────────────────────────────────────────────

def test_generate_overfitting_data_shape():
    from labs.lab3_overfitting import generate_overfitting_data
    X, y = generate_overfitting_data(n_samples=100)
    assert X.shape == (100, 1)
    assert y.shape == (100, 1)


def test_generate_overfitting_data_shuffled():
    from labs.lab3_overfitting import generate_overfitting_data
    X, _ = generate_overfitting_data(n_samples=100)
    diffs = np.diff(X.flatten())
    assert not np.all(diffs > 0), 'Data should be shuffled'


def test_lab3_get_params_defaults(app):
    from labs.lab3_overfitting import Lab3Widget
    w = Lab3Widget()
    p = w.get_params()
    assert p['epochs'] == 200
    assert p['learning_rate'] == pytest.approx(0.001)
    assert p['l1_reg'] == pytest.approx(0.0)
    assert p['l2_reg'] == pytest.approx(0.0)
    assert p['advanced'] is False
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_data.py::test_generate_overfitting_data_shape -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'labs.lab3_overfitting'`

- [ ] **Step 3: labs/lab3_overfitting.py 구현**

```python
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
    Returns: {'histories': [h_underfit, h_good, h_overfit], 'models': [m, m, m]}
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
        model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse',
                      metrics=['mae'])
        cb = make_keras_callback(stage_label, epochs, should_stop, emit_progress)
        h = model.fit(X, y, epochs=epochs, validation_split=0.2,
                      callbacks=[cb], verbose=0)
        histories.append(h)
        models_trained.append(model)

    return {'histories': histories, 'models': models_trained, 'X': X, 'y': y}


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
            ('_epochs_input', 'Epochs', '200',
             '3개 모델 각각에 적용되는 에포크 수.'),
            ('_lr_input', 'Learning Rate', '0.001',
             'Adam optimizer learning rate.'),
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
        histories = results['histories']
        models = results['models']
        X, y = results['X'], results['y']
        names = _MODEL_NAMES[:len(models)]
        colors = _MODEL_COLORS[:len(models)]

        for row in self._axes:
            for ax in row:
                ax.clear()
                ax.set_facecolor('#12121e')

        x_plot = np.linspace(-3, 3, 200).reshape(-1, 1).astype(np.float32)
        idx_sort = np.argsort(X.flatten())

        # Plot 1: 예측 비교
        self._axes[0, 0].scatter(X, y, color='white', s=10, alpha=0.4, label='데이터')
        for m, name, color in zip(models, names, colors):
            y_p = m.predict(x_plot, verbose=0)
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
        self._axes[0, 1].set_title('Train vs Val Loss', color='white', fontsize=10)
        self._axes[0, 1].set_yscale('log')
        self._axes[0, 1].legend(fontsize=7)

        # Plot 3: 오차 분포
        for m, name, color in zip(models, names, colors):
            y_p = m.predict(X, verbose=0)
            errors = np.abs(y - y_p).flatten()
            self._axes[1, 0].hist(errors, bins=20, alpha=0.6, color=color, label=name)
        self._axes[1, 0].set_title('오차 분포 (MAE)', color='white', fontsize=10)
        self._axes[1, 0].legend(fontsize=8)

        # Plot 4: 성능 비교표
        self._axes[1, 1].axis('off')
        col_labels = ['모델', 'Train MSE', 'Val MSE']
        table_data = []
        for h, name in zip(histories, names):
            train_mse = h.history['loss'][-1]
            val_mse = h.history.get('val_loss', [float('nan')])[-1]
            table_data.append([name, f'{train_mse:.4f}', f'{val_mse:.4f}'])
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
            cell.set_text_props(color='white')
        self._axes[1, 1].set_title('성능 비교', color='white', fontsize=10)

        self._figure.tight_layout()
        self._canvas.draw()

    def save_outputs(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        path = os.path.join(OUTPUTS_DIR, '03_overfitting_results.png')
        self._figure.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        return path

    def _save(self):
        path = self.save_outputs()
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text(f'저장 완료: {os.path.basename(path)}')
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 14 PASSED

- [ ] **Step 5: Commit**

```bash
git add labs/lab3_overfitting.py tests/test_data.py
git commit -m "feat: add Lab3Widget — overfitting vs underfitting demo"
```

---

### Task 8: labs/lab4_pendulum.py — 진자 주기

**Files:**
- Create: `labs/lab4_pendulum.py`
- Modify: `tests/test_data.py` (Lab 4 테스트 추가)

- [ ] **Step 1: tests/test_data.py에 Lab 4 테스트 추가**

```python
# 파일 끝에 추가

# ── Lab 4 ────────────────────────────────────────────────────────────────────

def test_generate_pendulum_data_shape():
    from labs.lab4_pendulum import generate_pendulum_data
    X, y = generate_pendulum_data(n_samples=100)
    assert X.shape == (100, 2)
    assert y.shape == (100, 1)


def test_generate_pendulum_data_period_positive():
    from labs.lab4_pendulum import generate_pendulum_data
    _, y = generate_pendulum_data(n_samples=100)
    assert np.all(y > 0), 'Period must be positive'


def test_rk4_pendulum_returns_arrays():
    from labs.lab4_pendulum import rk4_pendulum
    import math
    t, theta, omega = rk4_pendulum(L=1.0, theta0_rad=0.1, gamma=0.0, t_max=5.0, dt=0.05)
    assert len(t) == len(theta) == len(omega)
    assert t[0] == pytest.approx(0.0)


def test_rk4_pendulum_small_angle_period():
    """Small angle period should be approx 2π√(L/g) = 2.007s for L=1.0."""
    from labs.lab4_pendulum import rk4_pendulum
    import math
    t, theta, omega = rk4_pendulum(L=1.0, theta0_rad=0.05, gamma=0.0,
                                    t_max=20.0, dt=0.01)
    expected_period = 2 * math.pi * math.sqrt(1.0 / 9.81)
    # Count zero crossings in omega to estimate period
    zero_crossings = np.where(np.diff(np.sign(omega)))[0]
    if len(zero_crossings) >= 2:
        estimated_period = 2 * (t[zero_crossings[1]] - t[zero_crossings[0]])
        assert abs(estimated_period - expected_period) < 0.05


def test_lab4_get_params_defaults(app):
    from labs.lab4_pendulum import Lab4Widget
    w = Lab4Widget()
    p = w.get_params()
    assert p['epochs'] == 500
    assert p['learning_rate'] == pytest.approx(0.001)
    assert p['gamma'] == pytest.approx(0.0)
    assert p['advanced'] is False
    assert set(p['lengths']) == {0.5, 1.0, 2.0}
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_data.py::test_generate_pendulum_data_shape -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'labs.lab4_pendulum'`

- [ ] **Step 3: labs/lab4_pendulum.py 구현**

```python
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
    n = int(t_max / dt)
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
    """T ≈ T0 * (1 + (1/16)θ₀² + (11/3072)θ₀⁴) for undamped pendulum."""
    T0 = 2 * math.pi * math.sqrt(L / G)
    k = math.sin(theta0_rad / 2)
    return T0 * (1 + k ** 2 / 4 + 9 * k ** 4 / 64)


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

    X, y = generate_pendulum_data(n_samples=2000, gamma=gamma)
    model = build_lab4_model()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse',
                  metrics=['mae'])
    cb = make_keras_callback('진자 주기 학습', epochs, should_stop, emit_progress)
    history = model.fit(X, y, epochs=epochs, validation_split=0.2,
                        batch_size=64, callbacks=[cb], verbose=0)

    # RK4 시뮬레이션 (L=1.0, theta=20도 기준)
    t, theta, omega = rk4_pendulum(1.0, 20 * math.pi / 180, gamma=gamma,
                                    t_max=10.0, dt=0.01)

    return {
        'model': model,
        'history': history,
        'rk4': {'t': t, 'theta': theta, 'omega': omega, 'L': 1.0, 'gamma': gamma},
        'lengths': params.get('lengths', _DEFAULT_LENGTHS),
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
        model = results['model']
        history = results['history']
        rk4 = results['rk4']
        lengths = results.get('lengths', _DEFAULT_LENGTHS)
        gamma = results.get('gamma', 0.0)

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor('#12121e')

        colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
        theta_range = np.linspace(5, 80, 60)

        # Subplot 1: 주기 예측 (NN vs 해석해)
        for L, color in zip(lengths, colors):
            # 해석해
            T_analytical = np.array([
                _analytical_period(L, th * math.pi / 180, gamma)
                for th in theta_range
            ])
            self._axes[0].plot(theta_range, T_analytical, '--', color=color,
                               linewidth=1.5, alpha=0.6, label=f'L={L}m 해석해')
            # NN 예측
            X_pred = np.column_stack([
                np.full(len(theta_range), L),
                theta_range,
            ]).astype(np.float32)
            T_pred = model.predict(X_pred, verbose=0).flatten()
            self._axes[0].plot(theta_range, T_pred, '-', color=color,
                               linewidth=2, label=f'L={L}m NN')
        self._axes[0].set_title('주기 예측: NN vs 해석해', color='white', fontsize=10)
        self._axes[0].set_xlabel('각도 (°)', color='white')
        self._axes[0].set_ylabel('주기 T (s)', color='white')
        self._axes[0].legend(fontsize=7)

        # Subplot 2: RK4 시뮬레이션
        t, theta, omega = rk4['t'], rk4['theta'], rk4['omega']
        theta_deg = np.degrees(theta)
        self._axes[1].plot(t, theta_deg, color='#4d96ff', linewidth=1.5)
        self._axes[1].set_title(
            f'RK4 시뮬레이션 (L={rk4["L"]}m, γ={gamma:.2f})', color='white', fontsize=10
        )
        self._axes[1].set_xlabel('시간 (s)', color='white')
        self._axes[1].set_ylabel('각도 (°)', color='white')

        # Inset: 위상 공간
        ax_inset = self._axes[1].inset_axes([0.55, 0.55, 0.42, 0.42])
        ax_inset.plot(theta_deg, np.degrees(omega), color='#ff6b6b', linewidth=1)
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
        path = os.path.join(OUTPUTS_DIR, '04_pendulum_results.png')
        self._figure.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self._figure.get_facecolor())
        return path

    def _save(self):
        path = self.save_outputs()
        mw = self.window()
        if hasattr(mw, 'set_status_text'):
            mw.set_status_text(f'저장 완료: {os.path.basename(path)}')
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 19 PASSED

- [ ] **Step 5: Commit**

```bash
git add labs/lab4_pendulum.py tests/test_data.py
git commit -m "feat: add Lab4Widget — pendulum period prediction with RK4 simulation"
```

---

### Task 9: main.py에 모든 Lab 연결 — 최종 통합

**Files:**
- Modify: `main.py` (register_lab 호출 추가)

- [ ] **Step 1: main.py의 main() 함수 수정**

`main.py`의 `main()` 함수를 아래와 같이 교체:

```python
def main():
    import matplotlib
    matplotlib.use('QtAgg')

    from labs.lab1_1d import Lab1Widget
    from labs.lab2_projectile import Lab2Widget
    from labs.lab3_overfitting import Lab3Widget
    from labs.lab4_pendulum import Lab4Widget

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()

    for LabCls in (Lab1Widget, Lab2Widget, Lab3Widget, Lab4Widget):
        window.register_lab(LabCls())

    window._switch(0)
    window.show()
    sys.exit(app.exec())
```

- [ ] **Step 2: 전체 앱 실행 확인**

```bash
python main.py
```

Expected:
- 사이드바 4개 Lab 버튼 표시
- 각 Lab 클릭 시 파라미터 패널 + 빈 그래프 영역 전환
- "학습 시작" 클릭 → 상태바에 진행상황 표시
- 학습 완료 → 그래프 렌더링
- "PNG 저장" 클릭 → `outputs/` 파일 생성 확인
- "중단" 버튼 동작 확인

- [ ] **Step 3: 전체 테스트 수트 통과 확인**

```bash
python -m pytest tests/ -v
```

Expected: 전체 PASSED (실패 없음)

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire all 4 labs into MainWindow — Physics NN Explorer complete"
```

---

## Self-Review

**스펙 커버리지 체크:**

| 스펙 요구사항 | 담당 Task |
|---|---|
| 사이드바 네비게이션 | Task 4 |
| QThread + progress bar | Task 3, 4 |
| FigureCanvasQTAgg 임베드 + PNG 저장 | Task 5~8 |
| 심화 체크박스 (각 Lab) | Task 5~8 |
| 다크 테마 | Task 4 |
| 텍스트 입력 파라미터 | Task 5~8 |
| 중단 버튼 (Keras callback) | Task 3 |
| 상태바 한 줄 | Task 4 |
| 혼합 언어 | Task 4~8 |
| Lab 3 순차 학습 | Task 7 |
| Lab 4 RK4 정적 플롯 | Task 8 |
| 입력값 유효성 검사 | Task 5~8 |
| NaN loss 에러 처리 | Task 3 |
| outputs/ 자동 생성 | Task 5~8 |

**타입/시그니처 일관성:**
- `TrainingWorker(experiment_fn, params)` — Task 3 정의, Task 5~8 모두 동일하게 사용 ✓
- `make_keras_callback(stage_label, total_epochs, should_stop, emit_progress)` — Task 3 정의, Task 5~8 사용 ✓
- `experiment_fn(params, should_stop, emit_progress) → dict` — 모든 Lab 일관 ✓
- `MainWindow.update_status(stage, epoch, total, loss)` / `set_status_text(text)` — Task 4 정의, Task 5~8 호출 ✓
- `_tooltip_btn` — Task 5에서 정의, Task 6~8이 `labs.lab1_1d`에서 import ✓
