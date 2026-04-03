import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QStatusBar, QProgressBar,
)
from PySide6.QtGui import QPalette, QColor

PALETTE = {
    'window':       '#1e1e2e',
    'window_text':  '#cdd6f4',
    'base':         '#181825',
    'alt_base':     '#313244',
    'text':         '#cdd6f4',
    'button':       '#313244',
    'button_text':  '#cdd6f4',
    'highlight':    '#7fb8ff',
    'hl_text':      '#1e1e2e',
    'sidebar':      '#1a1a2e',
}


def apply_dark_theme(app):
    app.setStyle('Fusion')
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(PALETTE['window']))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(PALETTE['window_text']))
    p.setColor(QPalette.ColorRole.Base,            QColor(PALETTE['base']))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(PALETTE['alt_base']))
    p.setColor(QPalette.ColorRole.Text,            QColor(PALETTE['text']))
    p.setColor(QPalette.ColorRole.Button,          QColor(PALETTE['button']))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(PALETTE['button_text']))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(PALETTE['highlight']))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(PALETTE['hl_text']))
    app.setPalette(p)


class _SidebarButton(QPushButton):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(58)
        self._title = title
        self._subtitle = subtitle
        self._refresh(False)

    def setChecked(self, checked):
        super().setChecked(checked)
        self._refresh(checked)

    def _refresh(self, active):
        border = '3px solid #7fb8ff' if active else '3px solid transparent'
        bg = '#2d2d5e' if active else 'transparent'
        color = '#ffffff' if active else '#888888'
        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-left: {border};
                background: {bg};
                color: {color};
                text-align: left;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: #252545; color: #cccccc; }}
        """)
        self.setText(f"{self._title}\n{self._subtitle}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Physics NN Explorer')
        self.setMinimumSize(1100, 700)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(150)
        sidebar.setStyleSheet(f'background: {PALETTE["sidebar"]};')
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        hdr = QLabel('🔭  Labs')
        hdr.setStyleSheet(
            'color: #7fb8ff; font-weight: bold; font-size: 13px;'
            'padding: 12px; border-bottom: 1px solid #333;'
        )
        sl.addWidget(hdr)

        self._btns = []
        for i, (icon, title, sub) in enumerate([
            ('🔬', 'Lab 1', '1D 함수 근사'),
            ('🚀', 'Lab 2', '포물선 운동'),
            ('⚖️', 'Lab 3', '과적합 분석'),
            ('🕰️', 'Lab 4', '진자 주기'),
        ]):
            btn = _SidebarButton(f'{icon} {title}', sub)
            btn.clicked.connect(lambda _, idx=i: self._switch(idx))
            sl.addWidget(btn)
            self._btns.append(btn)

        sl.addStretch()
        root.addWidget(sidebar)

        # ── Stack ─────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f'background: {PALETTE["window"]};')
        root.addWidget(self._stack)

        # ── Status bar ────────────────────────────────────────
        sb = QStatusBar()
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            'QProgressBar { background: #222; border: none; }'
            'QProgressBar::chunk { background: #4a7a4a; }'
        )
        self._status_lbl = QLabel('준비')
        self._status_lbl.setStyleSheet('color: #888; font-size: 11px; padding-right: 8px;')
        sb.addWidget(self._progress, 1)
        sb.addPermanentWidget(self._status_lbl)
        self.setStatusBar(sb)

        self._switch(0)

    def _switch(self, index):
        for i, btn in enumerate(self._btns):
            btn.setChecked(i == index)
        if self._stack.count() > index:
            self._stack.setCurrentIndex(index)

    def register_lab(self, widget):
        """LabWidget을 스택에 추가. Task 9에서 호출."""
        self._stack.addWidget(widget)

    def update_status(self, stage, epoch, total, loss):
        """TrainingWorker.progress signal 에 연결."""
        pct = int(epoch / total * 100) if total else 0
        self._progress.setValue(pct)
        self._status_lbl.setText(f'{stage} — Epoch {epoch}/{total} — loss: {loss:.4f}')

    def set_status_text(self, text):
        self._progress.setValue(0)
        self._status_lbl.setText(text)


def _configure_matplotlib_korean():
    """macOS/Windows/Linux 한국어 폰트 설정."""
    import matplotlib
    import matplotlib.font_manager as fm
    candidates = [
        'AppleGothic', 'Apple SD Gothic Neo',   # macOS
        'Malgun Gothic',                          # Windows
        'NanumGothic', 'NanumBarunGothic',        # Linux (나눔)
        'Noto Sans KR', 'Noto Sans CJK KR',      # Linux (Noto)
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


def main():
    import matplotlib
    matplotlib.use('QtAgg')
    _configure_matplotlib_korean()

    from labs.lab1_1d import Lab1Widget
    from labs.lab2_projectile import Lab2Widget
    from labs.lab3_overfitting import Lab3Widget
    from labs.lab4_pendulum import Lab4Widget

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()

    for LabCls in (Lab1Widget, Lab2Widget, Lab3Widget, Lab4Widget):
        window.register_lab(LabCls())

    window._switch(0)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
