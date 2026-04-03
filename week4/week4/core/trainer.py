import math
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication
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

    def wait(self, msecs=None):
        """Wait for the thread to finish and flush queued cross-thread signals.

        Note: processEvents() is re-entrant — do not call this method from
        within a Qt slot, as it may cause unexpected recursive event processing.
        """
        result = super().wait(msecs) if msecs is not None else super().wait()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        return result


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
