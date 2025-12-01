"""
AppController — мост между QML-интерфейсом и Python-логикой SmartFile Flow.

На первой итерации:
- хранит и отдает пути source/target
- обрабатывает нажатие кнопки «Сортировать» (пока как заглушка)

Дальше сюда добавим вызов FileSorter и обновление статистики/логов.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread
import sys

from PySide6.QtCore import QObject, Signal, Slot, Property, QUrl
from PySide6.QtGui import QDesktopServices

# Гарантируем доступ к backend-модулям (src) при запуске GUI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from file_sorter import FileSorter
from logger import get_logger
from config import get_config


class AppController(QObject):
    """Контроллер приложения для связи QML ↔ Python."""

    sourceDirChanged = Signal()
    targetDirChanged = Signal()
    sortingStarted = Signal()
    sortingFinished = Signal()
    totalChanged = Signal()
    sortedChanged = Signal()
    skippedChanged = Signal()
    failedChanged = Signal()
    mlInfoChanged = Signal()
    mlReadyChanged = Signal()
    mlEnabledChanged = Signal()
    copyFilesChanged = Signal()
    useMlChanged = Signal()
    conflictResolutionChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        project_root = Path(__file__).resolve().parents[2]
        # Значения по умолчанию такие же, как у CLI: data/raw → data/sorted
        self._source_dir = str(project_root / "data" / "raw")
        self._target_dir = str(project_root / "data" / "sorted")
        self._is_sorting = False
        self._logger = get_logger()
        # Статистика последнего запуска
        self._total = 0
        self._sorted = 0
        self._skipped = 0
        self._failed = 0
        # Статус ML-модели
        cfg = get_config()
        self._ml_enabled = bool(getattr(cfg.settings, "use_ml", False))
        self._ml_ready = False
        if self._ml_enabled:
            self._ml_info = "ML: включён, модель будет использована при сортировке"
        else:
            self._ml_info = "ML: выключен (используются только правила)"

        # Режимы работы
        self._copy_files = bool(getattr(cfg.settings, "copy_files", True))
        self._use_ml = self._ml_enabled
        self._conflict_resolution = getattr(cfg.settings, "conflict_resolution", "rename")

    # --- sourceDir ---
    def getSourceDir(self) -> str:  # type: ignore[override]
        return self._source_dir

    def setSourceDir(self, value: str) -> None:  # type: ignore[override]
        if value and value != self._source_dir:
            self._source_dir = value
            self.sourceDirChanged.emit()

    sourceDir = Property(str, fget=getSourceDir, fset=setSourceDir, notify=sourceDirChanged)

    # --- targetDir ---
    def getTargetDir(self) -> str:  # type: ignore[override]
        return self._target_dir

    def setTargetDir(self, value: str) -> None:  # type: ignore[override]
        if value and value != self._target_dir:
            self._target_dir = value
            self.targetDirChanged.emit()

    targetDir = Property(str, fget=getTargetDir, fset=setTargetDir, notify=targetDirChanged)

    # --- статистика последней сортировки (отдельные свойства) ---
    def getTotal(self) -> int:
        return self._total

    def getSorted(self) -> int:
        return self._sorted

    def getSkipped(self) -> int:
        return self._skipped

    def getFailed(self) -> int:
        return self._failed

    total = Property(int, fget=getTotal, notify=totalChanged)
    sorted = Property(int, fget=getSorted, notify=sortedChanged)
    skipped = Property(int, fget=getSkipped, notify=skippedChanged)
    failed = Property(int, fget=getFailed, notify=failedChanged)

    # --- информация о ML-модели ---
    def getMlInfo(self) -> str:
        return self._ml_info

    def getMlReady(self) -> bool:
        return self._ml_ready

    def getMlEnabled(self) -> bool:
        return getattr(self, "_ml_enabled", False)

    mlInfo = Property(str, fget=getMlInfo, notify=mlInfoChanged)
    mlReady = Property(bool, fget=getMlReady, notify=mlReadyChanged)
    mlEnabled = Property(bool, fget=getMlEnabled, notify=mlEnabledChanged)

    # --- режимы работы (copy/use_ml/conflict) ---
    def getCopyFiles(self) -> bool:
        return self._copy_files

    def getUseMl(self) -> bool:
        return self._use_ml

    def getConflictResolution(self) -> str:
        return self._conflict_resolution

    copyFiles = Property(bool, fget=getCopyFiles, notify=copyFilesChanged)
    useMl = Property(bool, fget=getUseMl, notify=useMlChanged)
    conflictResolution = Property(str, fget=getConflictResolution, notify=conflictResolutionChanged)

    # --- действия, связанные с файловой системой ---
    @Slot()
    def openTargetFolder(self) -> None:
        """Открыть папку назначения в проводнике."""
        if not self._target_dir:
            return
        url = QUrl.fromLocalFile(self._target_dir)
        QDesktopServices.openUrl(url)

    @Slot(bool)
    def setCopyFiles(self, value: bool) -> None:
        """Переключатель 'Копировать файлы'."""
        cfg = get_config()
        cfg.settings.copy_files = bool(value)
        self._copy_files = bool(value)
        self.copyFilesChanged.emit()

    @Slot(bool)
    def setUseMl(self, value: bool) -> None:
        """Переключатель 'Использовать ML'."""
        cfg = get_config()
        cfg.settings.use_ml = bool(value)
        self._use_ml = bool(value)
        self._ml_enabled = self._use_ml
        # Обновляем текст статуса
        if self._use_ml:
            self._ml_info = "ML: включён, модель будет использована при сортировке"
        else:
            self._ml_info = "ML: выключен (используются только правила)"
        self.useMlChanged.emit()
        self.mlEnabledChanged.emit()
        self.mlInfoChanged.emit()

    @Slot(str)
    def setConflictResolution(self, mode: str) -> None:
        """Выбор стратегии конфликтов: skip / overwrite / rename."""
        if mode not in ("skip", "overwrite", "rename"):
            return
        cfg = get_config()
        cfg.settings.conflict_resolution = mode
        self._conflict_resolution = mode
        self.conflictResolutionChanged.emit()

    # --- действия из QML ---
    @Slot()
    def sortFiles(self) -> None:
        """
        Обработчик кнопки «Сортировать».

        Запускает сортировку файлов в отдельном потоке, чтобы не блокировать UI.
        Пока без детального прогресса — только запуск и завершение.
        """
        if self._is_sorting:
            # Уже идёт сортировка — игнорируем повторный клик
            return

        self._is_sorting = True
        self.sortingStarted.emit()

        source = self._source_dir
        target = self._target_dir

        def worker() -> None:
            try:
                self._logger.info(f"GUI: старт сортировки из '{source}' в '{target}'")
                sorter = FileSorter(source, target)
                stats = sorter.sort_all(show_progress=False)
                # Сохраняем статистику для QML (если sort_all вернул dict)
                if isinstance(stats, dict):
                    self._total = int(stats.get("total", 0))
                    self._sorted = int(stats.get("sorted", 0))
                    self._skipped = int(stats.get("skipped", 0))
                    self._failed = int(stats.get("failed", 0))
                    self.totalChanged.emit()
                    self.sortedChanged.emit()
                    self.skippedChanged.emit()
                    self.failedChanged.emit()

                # Обновляем статус ML-модели
                try:
                    ml_info = sorter.ml_classifier.get_model_info()
                    status = ml_info.get("status", "not_trained")
                    model_type = ml_info.get("model_type", "unknown")
                    vocab_size = ml_info.get("vocabulary_size", 0)
                    vector_size = ml_info.get("vector_size", 0)

                    self._ml_ready = status == "trained"
                    if self._ml_ready:
                        self._ml_info = (
                            f"ML: {model_type}, словарь {vocab_size}, размер вектора {vector_size}"
                        )
                    else:
                        # Если ML включен, но модель не готова — показываем это явно
                        if self._ml_enabled:
                            self._ml_info = "ML: включён, но модель не загружена/не обучена"
                        else:
                            self._ml_info = "ML: выключен (используются только правила)"

                    self.mlEnabledChanged.emit()
                    self.mlReadyChanged.emit()
                    self.mlInfoChanged.emit()
                except Exception as ml_exc:  # pragma: no cover - защитный блок
                    self._logger.warning(f"GUI: не удалось получить информацию о ML-модели: {ml_exc}")
                self._logger.info("GUI: сортировка завершена")
            except Exception as exc:  # pragma: no cover - защитный блок
                self._logger.error(f"GUI: ошибка сортировки: {exc}")
            finally:
                self._is_sorting = False
                self.sortingFinished.emit()

        Thread(target=worker, daemon=True).start()


