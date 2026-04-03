from tensorflow import keras


def build_lab1_model():
    """[128, 128, 64] tanh. Input: (1,) → Output: (1,)."""
    return keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(128, activation='tanh'),
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
        sanitized = name.replace(' ', '_').replace('[', '_').replace(']', '_').replace(',', '_').strip('_')
        sanitized = '_'.join(part for part in sanitized.split('_') if part)
        layers = [keras.Input(shape=(1,))]
        for s in sizes:
            layers.append(keras.layers.Dense(s, activation='tanh'))
        layers.append(keras.layers.Dense(1, activation='linear'))
        models[name] = keras.Sequential(layers, name=sanitized)
    return models


def build_lab1_extreme_model():
    """[256, 256, 128, 64] tanh. Input: (1,) → Output: (1,)."""
    return keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(256, activation='tanh'),
        keras.layers.Dense(128, activation='tanh'),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(1, activation='linear'),
    ], name='lab1_extreme')


def build_lab2_model():
    """[128, 64, 32] relu+dropout. Input: (3,) [v0, theta_deg, t] → Output: (2,) [x, y]."""
    return keras.Sequential([
        keras.Input(shape=(3,)),
        keras.layers.Dense(128, activation='relu'),
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
        keras.Input(shape=(1,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ], name='underfit')

    goodfit = keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(1, activation='linear'),
    ], name='goodfit')

    overfit = keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ], name='overfit')

    return underfit, goodfit, overfit


def build_lab4_model():
    """[64, 32, 16] relu+dropout. Input: (2,) [L_m, theta0_deg] → Output: (1,) [T_sec]."""
    return keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear'),
    ], name='lab4_model')
