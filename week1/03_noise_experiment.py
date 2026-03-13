import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import tensorflow as tf
import os

os.makedirs('outputs', exist_ok=True)

# ─── 실험 설정 ────────────────────────────────────────────────────────────────
NOISE_SCALES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
TRUE_W, TRUE_B = 2.0, -1.0
EPOCHS = 500
RANDOM_SEED = 42

X_base = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_clean = TRUE_W * X_base + TRUE_B   # [-3, -1, 1, 3, 5, 7]


# ─── 헬퍼 함수 ────────────────────────────────────────────────────────────────
def linear_function(x, w, b):
    return w * x + b


def run_experiment(scale):
    """주어진 scale로 노이즈를 만들고 3가지 방법으로 학습, (w, b) 반환."""
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=scale, size=len(X_base))
    y_noisy = y_clean + noise

    # Method 1: Neural Network
    tf.random.set_seed(RANDOM_SEED)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    history = model.fit(X_base, y_noisy, epochs=EPOCHS, verbose=0)
    weights = model.get_weights()
    w_nn = float(weights[0][0][0])
    b_nn = float(weights[1][0])
    final_loss = history.history['loss'][-1]

    # Method 2: NumPy Polyfit
    coef = np.polyfit(X_base, y_noisy, deg=1)
    w_poly, b_poly = float(coef[0]), float(coef[1])

    # Method 3: SciPy Curve Fit
    popt, _ = curve_fit(linear_function, X_base, y_noisy, p0=[0.5, 0.5])
    w_scipy, b_scipy = float(popt[0]), float(popt[1])

    return {
        'scale':  scale,
        'y_noisy': y_noisy,
        'nn':     (w_nn,    b_nn,    final_loss),
        'poly':   (w_poly,  b_poly),
        'scipy':  (w_scipy, b_scipy),
    }


# ─── 실험 실행 ────────────────────────────────────────────────────────────────
print("=" * 60)
print("노이즈 크기(scale) 변화 실험: y = 2x - 1")
print("=" * 60)
print(f"{'scale':>6}  {'NN w':>8} {'NN b':>8}  {'Poly w':>8} {'Poly b':>8}  {'Scipy w':>8} {'Scipy b':>8}")
print("-" * 75)

results = []
for scale in NOISE_SCALES:
    r = run_experiment(scale)
    results.append(r)
    w_nn, b_nn, loss = r['nn']
    w_p,  b_p        = r['poly']
    w_s,  b_s        = r['scipy']
    print(f"{scale:>6.1f}  {w_nn:>8.4f} {b_nn:>8.4f}  {w_p:>8.4f} {b_p:>8.4f}  {w_s:>8.4f} {b_s:>8.4f}")

print("-" * 75)
print(f"{'truth':>6}  {'2.0000':>8} {'-1.0000':>8}  {'2.0000':>8} {'-1.0000':>8}  {'2.0000':>8} {'-1.0000':>8}")


# ─── 그림 1: 각 scale별 피팅 결과 (서브플롯) ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Effect of Noise Scale on Fitting Methods\n(True: y = 2x - 1)', fontsize=14)
x_plot = np.linspace(-2, 5, 100)

for ax, r in zip(axes.flat, results):
    scale = r['scale']
    w_nn, b_nn, _ = r['nn']
    w_p,  b_p     = r['poly']
    w_s,  b_s     = r['scipy']

    ax.scatter(X_base, r['y_noisy'], color='red', s=60, zorder=5, label='Noisy data')
    ax.plot(x_plot, TRUE_W * x_plot + TRUE_B,  'k:', alpha=0.5, label='True (y=2x-1)')
    ax.plot(x_plot, w_nn * x_plot + b_nn,  'b-',  label=f'NN     w={w_nn:.2f}, b={b_nn:.2f}')
    ax.plot(x_plot, w_p  * x_plot + b_p,   'g--', label=f'Poly   w={w_p:.2f}, b={b_p:.2f}')
    ax.plot(x_plot, w_s  * x_plot + b_s,   'm-.', label=f'SciPy  w={w_s:.2f}, b={b_s:.2f}')
    ax.set_title(f'scale = {scale}')
    ax.set_xlim(-2, 5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path1 = 'outputs/03_noise_fit_comparison.png'
plt.savefig(path1, dpi=120)
print(f"\n[그림 1] 피팅 결과 비교 저장: {path1}")


# ─── 그림 2: scale vs |w 오차|, |b 오차| ─────────────────────────────────────
scales_arr = np.array(NOISE_SCALES)

w_err_nn   = np.abs([r['nn'][0]    - TRUE_W for r in results])
b_err_nn   = np.abs([r['nn'][1]    - TRUE_B for r in results])
w_err_poly = np.abs([r['poly'][0]  - TRUE_W for r in results])
b_err_poly = np.abs([r['poly'][1]  - TRUE_B for r in results])
w_err_sc   = np.abs([r['scipy'][0] - TRUE_W for r in results])
b_err_sc   = np.abs([r['scipy'][1] - TRUE_B for r in results])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Estimation Error vs Noise Scale', fontsize=13)

for ax, (e_nn, e_poly, e_sc, param) in zip(
        [ax1, ax2],
        [(w_err_nn, w_err_poly, w_err_sc, 'w (slope)'),
         (b_err_nn, b_err_poly, b_err_sc, 'b (intercept)')]):
    ax.plot(scales_arr, e_nn,   'bo-', label='Neural Network')
    ax.plot(scales_arr, e_poly, 'gs--', label='NumPy Polyfit')
    ax.plot(scales_arr, e_sc,   'm^-.', label='SciPy curve_fit')
    ax.set_xlabel('Noise scale')
    ax.set_ylabel(f'|estimated {param} − true {param}|')
    ax.set_title(f'{param} Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path2 = 'outputs/03_noise_error_plot.png'
plt.savefig(path2, dpi=120)
print(f"[그림 2] 오차 변화 그래프 저장: {path2}")

print("\n실험 완료!")
print("  - outputs/03_noise_fit_comparison.png : scale별 피팅 시각화")
print("  - outputs/03_noise_error_plot.png     : scale vs 오차 그래프")
