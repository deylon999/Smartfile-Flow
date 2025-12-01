"""
Скрипт для запуска всех тестов
"""
import sys
import unittest
from pathlib import Path

# Добавляем src в путь (utils находится на уровень выше src)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_all_tests():
    """Запускает все тесты"""
    # Находим все тестовые файлы
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent.parent / "tests"
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Возвращаем код выхода
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())

