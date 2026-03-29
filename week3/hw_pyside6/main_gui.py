import sys
import os
import matplotlib
matplotlib.use('qtagg')

import matplotlib.font_manager as fm

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMessageBox
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

# 탭 모듈은 matplotlib 백엔드 설정 이후에 임포트
from tabs.tab_perceptron import PerceptronTab
from tabs.tab_activation import ActivationTab
from tabs.tab_forward_prop import ForwardPropTab
from tabs.tab_mlp import MLPTab
from tabs.tab_universal import UniversalTab


def setup_korean_font():
    """시스템에서 사용 가능한 한글 폰트를 찾아 matplotlib에 설정"""
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
                 'Gulim', 'Batang', 'Dotum']:
        if font in font_list:
            matplotlib.rcParams['font.family'] = font
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Week 3: 신경망 기초 — PySide6 GUI')
        self.setMinimumSize(1000, 700)
        self.resize(1280, 820)
        self._init_menu()
        self._init_tabs()

    def _init_menu(self):
        menubar = self.menuBar()

        # ── 파일 메뉴 ──
        file_menu = menubar.addMenu('파일(&F)')

        save_action = QAction('현재 탭 저장(&S)', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('현재 탭의 그래프를 PNG로 저장합니다')
        save_action.triggered.connect(self.save_current_tab)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('종료(&Q)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('프로그램을 종료합니다')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ── 도움말 메뉴 ──
        help_menu = menubar.addMenu('도움말(&H)')

        about_action = QAction('정보(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _init_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(PerceptronTab(),  '퍼셉트론')
        self.tab_widget.addTab(ActivationTab(),  '활성화 함수')
        self.tab_widget.addTab(ForwardPropTab(), '순전파')
        self.tab_widget.addTab(MLPTab(),         'MLP / XOR')
        self.tab_widget.addTab(UniversalTab(),   '보편 근사')
        self.setCentralWidget(self.tab_widget)

    def save_current_tab(self):
        current = self.tab_widget.currentWidget()
        if hasattr(current, 'save_figure'):
            current.save_figure()
        else:
            QMessageBox.information(self, '저장', '저장할 그래프가 없습니다.')

    def show_about(self):
        QMessageBox.about(
            self,
            'Week 3 — 신경망 기초 정보',
            '<b>Week 3: 신경망 기초</b><br>'
            'PySide6 GUI 버전<br><br>'
            '<b>포함 모듈:</b><br>'
            '&nbsp;① 퍼셉트론 (AND / OR / XOR 게이트)<br>'
            '&nbsp;② 활성화 함수 비교<br>'
            '&nbsp;③ 순전파 (Forward Propagation)<br>'
            '&nbsp;④ MLP / XOR 학습 (Backpropagation)<br>'
            '&nbsp;⑤ 보편 근사 정리<br><br>'
            'Python 3.11 &nbsp;|&nbsp; PySide6 6.11'
        )


if __name__ == '__main__':
    setup_korean_font()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
