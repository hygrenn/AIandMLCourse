# Physics NN Explorer — Week 4

Neural Network로 물리 데이터를 학습·예측하는 PySide6 GUI 앱입니다.
4개의 Lab을 사이드바로 전환하며 파라미터를 조절하고 학습 결과를 실시간으로 확인할 수 있습니다.

---

## 실행 방법

### 1. 의존성 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip 사용
pip install pyside6 tensorflow numpy matplotlib scipy
```

### 2. 앱 실행

```bash
python main.py
```

---

## 화면 구성

```
┌──────────┬────────────────────────────────────────────┐
│ 🔭 Labs  │  Lab 화면 (파라미터 패널 + 그래프 영역)      │
│          │                                            │
│ ▶ Lab 1  │  [파라미터 입력]  │  [matplotlib 그래프]    │
│   Lab 2  │  Epochs          │                        │
│   Lab 3  │  Learning Rate   │                        │
│   Lab 4  │                  │                        │
│          │  [심화 과제 체크]  │                        │
│          │  [▶ 학습 시작]   │                        │
│          │  [■ 중단]        │                        │
├──────────┴────────────────────────────────────────────┤
│ ████████░░░  Epoch 150/200 — loss: 0.0123             │
└───────────────────────────────────────────────────────┘
```

- **사이드바**: Lab 탭 전환
- **파라미터 패널**: Epochs, Learning Rate 등 입력. `?` 버튼으로 툴팁 확인
- **그래프 영역**: 학습 완료 후 결과 자동 표시. 확대·이동 툴바 제공
- **상태바**: 학습 진행률(progress bar) + 현재 epoch/loss 표시

---

## Lab 별 설명

### Lab 1 — 1D 함수 근사

**Universal Approximation Theorem 실험**

Neural Network가 임의의 1D 함수를 얼마나 잘 근사하는지 확인합니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| 함수 선택 | sin(x) / cos(x)+0.5sin(2x) / x·sin(x) | sin(x) |
| Epochs | 학습 반복 횟수 | 3000 |
| Learning Rate | Adam optimizer 학습률 | 0.001 |

**그래프 (3개 subplot):**
1. 선택 함수 근사 결과 (실제값 vs NN 예측)
2. Small / Medium / Large / Very Large 네트워크 크기 비교
3. 극한 복잡도 함수 테스트

**심화 과제 체크 시:** `tanh(x)`, `x³` 함수 추가 학습 — 그래프 하단에 2개 subplot이 추가됩니다.

---

### Lab 2 — 포물선 운동

**NN으로 2D 물리 궤적 예측**

입력 (v₀, θ, t) → 출력 (x, y) 구조로, 포물선 운동 방정식을 데이터로부터 학습합니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| Epochs | 학습 반복 횟수 | 500 |
| Learning Rate | Adam optimizer 학습률 | 0.001 |

**그래프 (2개 subplot):**
1. 3가지 조건(v₀=20/30/40, θ=30°/45°/60°)의 실제 궤적 vs NN 예측 비교
2. 학습 곡선 (Train / Validation Loss)

**심화 과제 체크 시:** 공기 저항(drag coefficient) 입력 필드가 나타납니다. 제목에 `(공기 저항 drag=X.XX)` 가 표시됩니다.

---

### Lab 3 — 과적합 vs 과소적합

**모델 복잡도가 성능에 미치는 영향 시연**

Underfit / Good Fit / Overfit 3가지 모델을 순차적으로 학습해 비교합니다.

| 모델 | 구조 | 특징 |
|------|------|------|
| Underfit | [4] | 너무 단순, 패턴 학습 불가 |
| Good Fit | [32, 16] + Dropout | 적절한 일반화 |
| Overfit | [256, 128, 64, 32] | 훈련 데이터 노이즈까지 학습 |

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| Epochs | 3개 모델 각각에 적용 | 200 |
| Learning Rate | Adam optimizer 학습률 | 0.001 |

**그래프 (2×2, 4개 subplot):**
1. 3개 모델 예측 곡선 비교
2. Train vs Validation Loss (학습 곡선)
3. 오차 분포 (MAE 히스토그램)
4. Train MSE / Val MSE 성능 비교표

**심화 과제 체크 시:** L1/L2 regularization λ 입력 — Good Fit 모델에 적용. 학습 곡선 제목에 `[L1=X, L2=X]` 가 표시됩니다.

---

### Lab 4 — 진자 주기

**비선형 진자 운동 학습 + RK4 수치 시뮬레이션**

진자 길이(L)와 초기 각도(θ₀)를 입력받아 주기 T를 예측합니다.

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| 진자 길이 | L=0.5m / 1.0m / 2.0m 체크박스 | 1.0m |
| 각도 범위 | 최소/최대 각도 (°) | 5° ~ 80° |
| Epochs | 학습 반복 횟수 | 500 |
| Learning Rate | Adam optimizer 학습률 | 0.001 |

**그래프 (3개 subplot):**
1. 길이별 주기 예측: NN vs 해석해(타원 적분 근사) 비교
2. RK4 수치 시뮬레이션: 각도-시간 그래프 + 위상 공간(θ vs ω) inset
3. 학습 곡선 (Train / Validation Loss)

**심화 과제 체크 시:** 감쇠 계수 γ 입력 — 운동 방정식 `d²θ/dt² = -(g/L)sin(θ) - γ·dθ/dt` 적용. 주기 예측 제목에 `(감쇠 γ=X.XX)` 가 표시됩니다.

---

## PNG 저장

학습 완료 후 그래프 영역 하단의 **💾 PNG 저장** 버튼을 클릭하면 `outputs/` 디렉토리에 자동 저장됩니다.

| Lab | 일반 모드 | 심화 과제 모드 |
|-----|-----------|----------------|
| Lab 1 | `outputs/01_lab1_results.png` | `outputs/01_lab1_results_advanced.png` |
| Lab 2 | `outputs/02_projectile_results.png` | `outputs/02_projectile_results_advanced.png` |
| Lab 3 | `outputs/03_overfitting_results.png` | `outputs/03_overfitting_results_advanced.png` |
| Lab 4 | `outputs/04_pendulum_results.png` | `outputs/04_pendulum_results_advanced.png` |

심화 과제를 활성화한 채로 학습하면 파일명에 `_advanced` 가 붙어 별도로 저장됩니다.

---

## 프로젝트 구조

```
week4/
├── main.py                   # 앱 진입점 (MainWindow, 사이드바)
├── core/
│   ├── models.py             # Keras 모델 팩토리 함수
│   └── trainer.py            # TrainingWorker (QThread 기반 백그라운드 학습)
├── labs/
│   ├── lab1_1d.py            # Lab 1 위젯
│   ├── lab2_projectile.py    # Lab 2 위젯
│   ├── lab3_overfitting.py   # Lab 3 위젯
│   └── lab4_pendulum.py      # Lab 4 위젯
├── outputs/                  # PNG 저장 디렉토리 (자동 생성)
├── tests/                    # pytest 테스트 코드
└── pyproject.toml            # 의존성 정의
```

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| GUI | PySide6 (Qt6) — Fusion 다크 테마 |
| 딥러닝 | TensorFlow / Keras |
| 그래프 | matplotlib (FigureCanvasQTAgg 임베드) |
| 수치 계산 | NumPy, SciPy (RK4) |
| 학습 스레드 | QThread — UI 블로킹 없이 백그라운드 학습 |
