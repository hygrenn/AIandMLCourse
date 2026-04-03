# Week 4 PySide6 앱 설계 문서

**날짜**: 2026-04-01
**프로젝트**: Physics NN Explorer
**목적**: week4.md의 Lab 1~4 + 기본/심화 과제를 PySide6 GUI 앱으로 구현

---

## 1. 개요

Neural Network로 물리 데이터를 학습/예측하는 4개 Lab을 하나의 PySide6 앱으로 통합한다. 각 Lab은 독립 위젯으로 구현되며, 공통 학습 인프라(QThread)를 공유한다.

### 확정된 요구사항

| 항목 | 결정 |
|------|------|
| 레이아웃 | 사이드바 네비게이션 (단일 창) |
| 학습 실행 | QThread 백그라운드 + progress bar + 상태바 |
| 그래프 표시 | FigureCanvasQTAgg 임베드 + PNG 자동 저장 |
| 심화 과제 | 각 Lab UI 내 체크박스로 활성화 |
| 테마 | 다크 (Fusion 스타일 + 커스텀 QPalette) |
| 파라미터 입력 | 텍스트 입력 필드 (epochs, learning rate, 조건 선택) |
| 중단 버튼 | 있음 (Keras callback 기반) |
| 로그/콘솔 | 상태바 한 줄 (epoch/loss) |
| UI 언어 | 혼합 (한국어 레이블 + 영어 기술용어) |
| 학습 아키텍처 고정 | Layer 구조 고정, epochs + learning rate만 조절 |
| Lab 3 학습 방식 | 3개 모델 순차 학습 |
| Lab 4 RK4 표시 | 정적 플롯 |
| 이전 결과 처리 | 덮어쓰기 |
| 앱 이름 | Physics NN Explorer |
| 이론 설명 | 각 파라미터 옆 `?` 툴팁 |

---

## 2. 파일 구조

```
week4/
├── main.py                    # QApplication, MainWindow, 사이드바
├── labs/
│   ├── __init__.py
│   ├── lab1_1d.py             # Lab1Widget — 1D 함수 근사
│   ├── lab2_projectile.py     # Lab2Widget — 포물선 운동
│   ├── lab3_overfitting.py    # Lab3Widget — 과적합 vs 과소적합
│   └── lab4_pendulum.py       # Lab4Widget — 진자 주기
├── core/
│   ├── __init__.py
│   ├── trainer.py             # TrainingWorker(QThread)
│   └── models.py              # Keras 모델 팩토리 함수
├── outputs/                   # PNG 자동 저장 디렉토리
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-01-week4-pyside6-design.md
└── pyproject.toml
```

---

## 3. UI 구조

### 메인 창 레이아웃

```
┌─────────────────────────────────────────────────────────┐
│ Physics NN Explorer                              ▢  ✕   │
├──────────────┬──────────────────────────────────────────┤
│ 🔭 Labs      │ Lab 1: 1D 함수 근사                       │
│              │ Universal Approximation Theorem 실험       │
│ ▶ Lab 1      ├────────────────┬────────────────────────  │
│   Lab 2      │ 파라미터       │                          │
│   Lab 3      │                │   matplotlib 그래프      │
│   Lab 4      │ 함수 선택      │   (FigureCanvasQTAgg)    │
│              │ Epochs         │                          │
│              │ Learning Rate  │                          │
│              │                │                          │
│              │ [심화 과제]    │                          │
│              │ □ 추가 함수    │                          │
│              │                │                          │
│              │ [▶ 학습 시작]  ├────────────────────────  │
│              │ [■ 중단]       │ 🔍 확대   💾 PNG 저장    │
├──────────────┴──────────────────────────────────────────┤
│ ████████████████░░░░░  Epoch 1950/3000 — loss: 0.0042   │
└─────────────────────────────────────────────────────────┘
```

### 사이드바
- 너비 150px, 다크 배경 (`#1a1a2e`)
- 현재 선택 Lab: 좌측 파란 테두리 + 배경 하이라이트
- Lab 이름 + 한국어 부제목

### 파라미터 패널
- 너비 200px
- 각 파라미터: 레이블(한국어) + `?` 툴팁 버튼 + 입력 위젯
- 심화 과제 체크박스: 체크 시 추가 입력 필드 동적 표시
- 하단 고정: "학습 시작" 버튼(초록), "중단" 버튼(빨강)

### 그래프 영역
- `FigureCanvasQTAgg` + `NavigationToolbar2QT` (확대/축소/이동 내장)
- 하단 별도 "PNG 저장" 버튼 → `outputs/` 저장
- matplotlib 스타일: `dark_background`

### 상태바
- Progress bar (QProgressBar)
- 현재 상태 텍스트: "Epoch N/M — loss: X.XXXX" 또는 "학습 완료" 또는 "중단됨"

---

## 4. 컴포넌트 설계

### TrainingWorker(QThread) — `core/trainer.py`

```python
class TrainingWorker(QThread):
    progress = Signal(int, int, float)   # epoch, total, loss
    finished = Signal(object, object)    # history, model
    error = Signal(str)                  # error message

    def __init__(self, model_fn, data_fn, params): ...
    def run(self): ...          # Keras 학습 실행
    def stop(self): ...         # 중단 플래그 설정
```

- `model_fn`: `core/models.py`의 팩토리 함수
- `data_fn`: 각 Lab의 데이터 생성 함수
- Keras `LambdaCallback`으로 매 epoch마다 progress signal 발생 + 중단 플래그 확인

### LabWidget 공통 인터페이스

```python
class LabWidget(QWidget):
    def get_params(self) -> dict: ...          # 파라미터 수집 + 유효성 검사
    def render_results(self, history, model): ... # 그래프 업데이트
    def save_outputs(self): ...                # PNG 저장
```

### core/models.py — Keras 모델 팩토리

각 Lab의 모델을 생성하는 함수 모음:
- `build_lab1_model(hidden_layers)` → Sequential (tanh 활성화)
- `build_lab2_model()` → Sequential (relu + dropout)
- `build_lab3_models()` → (underfit, goodfit, overfit) 3개 반환
- `build_lab4_model()` → Sequential (relu + dropout)

---

## 5. Lab별 상세 스펙

### Lab 1 — 1D 함수 근사

**파라미터:**
- 함수 선택 (드롭다운): `sin(x)` / `cos(x)+0.5sin(2x)` / `x·sin(x)`
- Epochs (기본값: 3000)
- Learning Rate (기본값: 0.001)

**심화 체크박스:** "추가 함수 포함 (tanh(x), x³)"
- 체크 시: 전체 함수 세트로 학습, 극한 복잡도 테스트 포함

**그래프 (3개 subplot):**
1. 기본 함수 근사 결과 (실제값 vs 예측값)
2. 네트워크 크기 비교 (Small/Medium/Large/Very Large)
3. 극한 복잡도 테스트

**PNG 저장:** `outputs/01_1d_approximation.png`, `outputs/01_network_comparison.png`, `outputs/01_extreme_test.png`

---

### Lab 2 — 포물선 운동

**파라미터:**
- 테스트 조건 선택 (드롭다운): `v₀=20m/s θ=30°` / `v₀=30m/s θ=45°` / `v₀=40m/s θ=60°` / 전체
- Epochs (기본값: 500)
- Learning Rate (기본값: 0.001)

**심화 체크박스:** "공기 저항 포함"
- 체크 시: drag coefficient 입력 필드 추가 (기본값: 0.1)

**그래프 (2개 subplot):**
1. 3가지 조건 궤적 비교 (실제 물리 vs 예측)
2. 학습 곡선 + 각도/속도별 오차 분석

**PNG 저장:** `outputs/02_projectile_trajectories.png`, `outputs/02_projectile_analysis.png`

---

### Lab 3 — 과적합 vs 과소적합

**파라미터:**
- Epochs (기본값: 200)
- Learning Rate (기본값: 0.001)

**심화 체크박스:** "L1/L2 Regularization"
- 체크 시: λ 값 입력 필드 추가 (기본값: 0.01), Good Fit 모델에 적용

**학습 방식:** Underfit → Good Fit → Overfit 순차 학습
- Lab3Widget이 3개의 `TrainingWorker`를 순서대로 생성/실행 (worker1 finished signal → worker2 start → worker3 start)
- 상태바: "1/3 모델 학습 중 (Underfit)...", "2/3 모델 학습 중 (Good Fit)..."
- 3개 모두 완료 후 `render_results()`에 3개 history 전달

**그래프 (4개 subplot):**
1. 3개 모델 예측 비교
2. Train vs Validation 학습 곡선
3. 오차 분석
4. 성능 비교표

**PNG 저장:** `outputs/03_overfitting_comparison.png`, `outputs/03_training_curves.png`, `outputs/03_error_analysis.png`, `outputs/03_comparison_table.png`

---

### Lab 4 — 진자 주기

**파라미터:**
- 진자 길이 선택 (체크박스 멀티셀렉트): `L=0.5m` / `L=1.0m` / `L=2.0m`
- 각도 범위: 최소/최대 입력 (기본: 5° ~ 80°)
- Epochs (기본값: 500)
- Learning Rate (기본값: 0.001)

**심화 체크박스:** "감쇠 진자 (Damped Pendulum)"
- 체크 시: 감쇠 계수 γ 입력 필드 추가 (기본값: 0.1)
- 운동 방정식: `d²θ/dt² = -(g/L)sin(θ) - γ·dθ/dt`

**그래프 (3개 subplot):**
1. 길이별 주기 예측 (NN 예측 vs 해석해)
2. RK4 시뮬레이션 (각도 vs 시간, 위상 공간)
3. 학습 곡선 + 길이/각도별 오차 분석

**PNG 저장:** `outputs/04_pendulum_prediction.png`, `outputs/04_pendulum_simulation.png`, `outputs/04_pendulum_analysis.png`

---

## 6. 데이터 흐름

### 학습 실행
```
사용자 "▶ 학습 시작" 클릭
  → LabWidget.get_params() — 입력 유효성 검사
  → TrainingWorker(model_fn, data_fn, params) 생성
  → worker.start() — QThread 시작
  → progress signal → 상태바 업데이트 (메인 스레드)
  → finished signal → render_results() + save_outputs()
  → 상태바: "학습 완료 — outputs/ 저장됨"
```

### 중단
```
사용자 "■ 중단" 클릭
  → worker.stop() — 플래그 설정
  → Keras LambdaCallback — 다음 epoch 전 플래그 확인 → 조기 종료
  → finished signal (부분 결과)
  → 상태바: "학습 중단됨 (Epoch N에서)"
```

---

## 7. 에러 처리

| 상황 | 처리 |
|------|------|
| 잘못된 입력값 (epochs < 1, lr ≤ 0) | 학습 시작 전 QMessageBox 경고 |
| 학습 중 NaN loss 감지 | error signal → "학습 발산. Learning Rate를 낮춰보세요" 다이얼로그 |
| outputs/ 디렉토리 없음 | 자동 생성 |
| 학습 중 재실행 시도 | "학습 시작" 버튼 비활성화로 방지 |

---

## 8. 테마 및 스타일

- `QApplication.setStyle("Fusion")`
- 커스텀 `QPalette`: 배경 `#1e1e2e`, 사이드바 `#1a1a2e`, 텍스트 `#cdd6f4`
- matplotlib: `plt.style.use('dark_background')`
- 강조색: 파란색 `#7fb8ff` (선택 항목), 초록 `#4a7a4a` (실행), 빨강 `#7a4a4a` (중단)

---

## 9. 의존성

```toml
[project]
dependencies = [
    "pyside6",
    "tensorflow",
    "numpy",
    "matplotlib",
    "scipy",          # RK4 시뮬레이션 (Lab 4)
]
```
