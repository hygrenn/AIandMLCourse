import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope='session')
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance
