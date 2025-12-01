"""
Тесты для FileSorter
"""
import sys
import tempfile
import shutil
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from file_sorter import FileSorter


class TestFileSorter(unittest.TestCase):
    """Тесты для FileSorter"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем временные директории
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.temp_dir / "source"
        self.target_dir = self.temp_dir / "target"
        
        self.source_dir.mkdir()
        self.target_dir.mkdir()
        
        self.sorter = FileSorter(str(self.source_dir), str(self.target_dir))
    
    def tearDown(self):
        """Очистка после каждого теста"""
        # Удаляем временные директории
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_extract_text_from_txt(self):
        """Тест извлечения текста из .txt файла"""
        # Создаем тестовый файл
        test_file = self.source_dir / "test.txt"
        test_file.write_text("Это тестовый текст для проверки", encoding='utf-8')
        
        text = self.sorter.extract_text_from_file(test_file)
        
        self.assertIsNotNone(text, "Текст должен быть извлечен")
        self.assertIn("тестовый", text.lower(), "Текст должен содержать содержимое файла")
    
    def test_extract_text_from_empty_file(self):
        """Тест извлечения текста из пустого файла"""
        test_file = self.source_dir / "empty.txt"
        test_file.write_text("", encoding='utf-8')
        
        text = self.sorter.extract_text_from_file(test_file)
        
        self.assertEqual(text, "", "Пустой файл должен вернуть пустую строку")
    
    def test_extract_text_from_nonexistent_file(self):
        """Тест извлечения текста из несуществующего файла"""
        test_file = self.source_dir / "nonexistent.txt"
        
        text = self.sorter.extract_text_from_file(test_file)
        
        self.assertIsNone(text, "Несуществующий файл должен вернуть None")
    
    def test_categorize_with_rules(self):
        """Тест категоризации по правилам"""
        text = "Это рабочий проект с задачами и дедлайнами"
        category, confidence = self.sorter.categorize_with_rules(text)
        
        self.assertIsNotNone(category, "Категория должна быть определена")
        self.assertGreaterEqual(confidence, 0.0, "Уверенность должна быть >= 0")
    
    def test_categorize_with_rules_empty_text(self):
        """Тест категоризации пустого текста"""
        category, confidence = self.sorter.categorize_with_rules("")
        
        self.assertEqual(category, 'other', "Пустой текст должен быть 'other'")
        self.assertEqual(confidence, 0.0, "Уверенность должна быть 0")
    
    def test_scan_directory(self):
        """Тест сканирования директории"""
        # Создаем тестовые файлы
        (self.source_dir / "test1.txt").write_text("test", encoding='utf-8')
        (self.source_dir / "test2.txt").write_text("test", encoding='utf-8')
        (self.source_dir / "test3.pdf").write_text("test", encoding='utf-8')
        
        files = self.sorter.scan_directory()
        
        self.assertGreaterEqual(len(files), 2, "Должно найти хотя бы 2 файла")
        # Проверяем, что найдены правильные файлы
        file_names = [f.name for f in files]
        self.assertIn("test1.txt", file_names)
        self.assertIn("test2.txt", file_names)
    
    def test_scan_empty_directory(self):
        """Тест сканирования пустой директории"""
        files = self.sorter.scan_directory()
        
        self.assertEqual(len(files), 0, "Пустая директория должна вернуть пустой список")
    
    def test_resolve_conflict_rename(self):
        """Тест разрешения конфликта с переименованием"""
        # Создаем существующий файл
        existing_file = self.target_dir / "work" / "test.txt"
        existing_file.parent.mkdir(exist_ok=True)
        existing_file.write_text("existing", encoding='utf-8')
        
        # Устанавливаем стратегию rename
        self.sorter.config.settings.conflict_resolution = 'rename'
        
        target_path = self.target_dir / "work" / "test.txt"
        resolved_path, should_process = self.sorter._resolve_conflict(target_path)
        
        self.assertTrue(should_process, "Должен обработать файл")
        self.assertNotEqual(resolved_path.name, "test.txt", 
                          "Файл должен быть переименован")
        self.assertIn("_1", resolved_path.name, "Имя должно содержать номер")
    
    def test_resolve_conflict_skip(self):
        """Тест разрешения конфликта с пропуском"""
        # Создаем существующий файл
        existing_file = self.target_dir / "work" / "test.txt"
        existing_file.parent.mkdir(exist_ok=True)
        existing_file.write_text("existing", encoding='utf-8')
        
        # Устанавливаем стратегию skip
        self.sorter.config.settings.conflict_resolution = 'skip'
        
        target_path = self.target_dir / "work" / "test.txt"
        resolved_path, should_process = self.sorter._resolve_conflict(target_path)
        
        self.assertFalse(should_process, "Файл должен быть пропущен")
    
    def test_resolve_conflict_overwrite(self):
        """Тест разрешения конфликта с перезаписью"""
        # Создаем существующий файл
        existing_file = self.target_dir / "work" / "test.txt"
        existing_file.parent.mkdir(exist_ok=True)
        existing_file.write_text("existing", encoding='utf-8')
        
        # Устанавливаем стратегию overwrite
        self.sorter.config.settings.conflict_resolution = 'overwrite'
        
        target_path = self.target_dir / "work" / "test.txt"
        resolved_path, should_process = self.sorter._resolve_conflict(target_path)
        
        self.assertTrue(should_process, "Должен обработать файл")
        self.assertEqual(resolved_path.name, "test.txt", 
                        "Имя должно остаться прежним")


if __name__ == '__main__':
    unittest.main()

