import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

class FileSorterLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Создаем имя файла с timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"file_sorter_{timestamp}.log"
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Настраивает логгер"""
        self.logger = logging.getLogger('FileSorter')
        self.logger.setLevel(logging.INFO)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Добавляем обработчики
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Логгер инициализирован. Файл: {self.log_file}")
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def start_session(self, source_dir: str, target_dir: str):
        """Логирует начало сессии"""
        self.logger.info("=" * 50)
        self.logger.info("🚀 ЗАПУСК СОРТИРОВКИ ФАЙЛОВ")
        self.logger.info(f"📁 Источник: {source_dir}")
        self.logger.info(f"🎯 Цель: {target_dir}")
        self.logger.info("=" * 50)
    
    def end_session(self, processed: int, total: int):
        """Логирует завершение сессии"""
        self.logger.info("=" * 50)
        self.logger.info(f"✅ СОРТИРОВКА ЗАВЕРШЕНА")
        self.logger.info(f"📊 Обработано: {processed}/{total} файлов")
        self.logger.info("=" * 50)

# Синглтон для простого использования
_logger_instance: Optional[FileSorterLogger] = None

def get_logger() -> FileSorterLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FileSorterLogger()
    return _logger_instance