"""
Точка входа для GUI-версии SmartFile Flow.

Минимальный запуск:
- создаёт Qt-приложение на PySide6
- загружает QML-файл gui/qml/main.qml
- показывает пустое окно
"""

from pathlib import Path
import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

# Гарантируем, что корень проекта есть в sys.path,
# чтобы можно было делать import gui.* даже при прямом запуске файла.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.core.app_controller import AppController


def run_gui() -> int:
    """Запускает GUI SmartFile Flow (PySide6 + QML)."""
    # Корень проекта = папка, где лежит main.py верхнего уровня
    project_root = PROJECT_ROOT
    qml_dir = project_root / "gui" / "qml"
    main_qml = qml_dir / "main.qml"

    if not main_qml.exists():
        # На ранней стадии разработки лучше явно падать с понятной ошибкой
        raise FileNotFoundError(f"Не найден QML-файл: {main_qml}")

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Регистрируем контроллер в контексте QML до загрузки main.qml
    controller = AppController()
    engine.rootContext().setContextProperty("appController", controller)

    engine.load(str(main_qml))

    if not engine.rootObjects():
        # Если QML не загрузился — корректно завершаемся с кодом ошибки
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())


