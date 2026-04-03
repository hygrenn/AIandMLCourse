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
    callback.set_model(mock_model)
    callback.on_epoch_end(0, {'loss': 0.1})
    assert mock_model.stop_training is True


def test_keras_callback_does_not_stop_when_flag_false():
    from core.trainer import make_keras_callback

    mock_model = MagicMock()
    mock_model.stop_training = False

    callback = make_keras_callback('테스트', 10, lambda: False, lambda *a: None)
    callback.set_model(mock_model)
    callback.on_epoch_end(0, {'loss': 0.1})
    assert mock_model.stop_training is False


def test_keras_callback_raises_on_nan_loss():
    from core.trainer import make_keras_callback

    mock_model = MagicMock()
    callback = make_keras_callback('테스트', 10, lambda: False, lambda *a: None)
    callback.set_model(mock_model)

    with pytest.raises(ValueError, match='NaN'):
        callback.on_epoch_end(0, {'loss': float('nan')})
