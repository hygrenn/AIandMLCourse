import numpy as np


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


def test_lab1_size_models_output_shapes():
    from core.models import build_lab1_size_models
    models = build_lab1_size_models()
    for name, model in models.items():
        model.compile(optimizer='adam', loss='mse')
        result = model.predict(np.array([[0.5]]), verbose=0)
        assert result.shape == (1, 1), f"Model '{name}' output shape mismatch"


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
