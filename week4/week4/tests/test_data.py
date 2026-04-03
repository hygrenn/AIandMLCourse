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
    assert not np.array_equal(X.flatten(), np.sort(X.flatten())), 'Data should be shuffled, not sorted'


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


# ── Lab 3 ────────────────────────────────────────────────────────────────────

def test_generate_overfitting_data_shape():
    from labs.lab3_overfitting import generate_overfitting_data
    X, y = generate_overfitting_data(n_samples=100)
    assert X.shape == (100, 1)
    assert y.shape == (100, 1)


def test_generate_overfitting_data_shuffled():
    from labs.lab3_overfitting import generate_overfitting_data
    X, _ = generate_overfitting_data(n_samples=100)
    assert not np.array_equal(X.flatten(), np.sort(X.flatten())), 'Data should be shuffled'


def test_lab3_get_params_defaults(app):
    from labs.lab3_overfitting import Lab3Widget
    w = Lab3Widget()
    p = w.get_params()
    assert p['epochs'] == 200
    assert p['learning_rate'] == pytest.approx(0.001)
    assert p['l1_reg'] == pytest.approx(0.0)
    assert p['l2_reg'] == pytest.approx(0.0)
    assert p['advanced'] is False


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
