"""
Тесты для Config
"""
import sys
import tempfile
import yaml
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from config import Config, SettingsConfig


class TestConfig(unittest.TestCase):
    """Тесты для Config"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
    
    def tearDown(self):
        """Очистка после каждого теста"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Тест создания конфига по умолчанию"""
        # Создаем конфиг без файла
        config = Config(str(self.config_path))
        
        self.assertIsNotNone(config.settings, "Настройки должны быть созданы")
        self.assertGreater(len(config.settings.supported_extensions), 0,
                          "Должны быть поддерживаемые расширения")
    
    def test_load_config_from_file(self):
        """Тест загрузки конфига из файла"""
        # Создаем тестовый конфиг
        test_config = {
            'categories': {
                'test': {
                    'keywords': [['тест', 1.0]],
                    'color': '🔵',
                    'description': 'Тестовая категория'
                }
            },
            'settings': {
                'supported_extensions': ['.txt'],
                'min_confidence_score': 2.0,
                'copy_files': False,
                'use_ml': True
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, allow_unicode=True)
        
        config = Config(str(self.config_path))
        
        self.assertIn('test', config.categories, "Категория должна быть загружена")
        self.assertEqual(config.settings.min_confidence_score, 2.0,
                        "Настройки должны быть загружены")
    
    def test_get_category_names(self):
        """Тест получения имен категорий"""
        config = Config(str(self.config_path))
        
        # Добавляем категории вручную для теста
        from config import CategoryConfig
        config.categories['test1'] = CategoryConfig("test1", [], "", "")
        config.categories['test2'] = CategoryConfig("test2", [], "", "")
        
        names = config.get_category_names()
        
        self.assertIn('test1', names)
        self.assertIn('test2', names)
        self.assertGreaterEqual(len(names), 2)
    
    def test_get_weighted_keywords(self):
        """Тест получения ключевых слов с весами"""
        # Создаем конфиг с категорией
        test_config = {
            'categories': {
                'work': {
                    'keywords': [['работа', 2.0], ['проект', 1.5]],
                    'color': '🔵',
                    'description': 'Работа'
                }
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, allow_unicode=True)
        
        config = Config(str(self.config_path))
        keywords = config.get_weighted_keywords('work')
        
        self.assertEqual(len(keywords), 2, "Должно быть 2 ключевых слова")
        self.assertEqual(keywords[0][1], 2.0, "Вес первого слова должен быть 2.0")
        self.assertEqual(keywords[1][1], 1.5, "Вес второго слова должен быть 1.5")
    
    def test_invalid_config_handling(self):
        """Тест обработки неверного конфига"""
        # Создаем неверный конфиг
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")
        
        # Не должно упасть с ошибкой
        config = Config(str(self.config_path))
        
        # Должны использоваться значения по умолчанию
        self.assertIsNotNone(config.settings)


if __name__ == '__main__':
    unittest.main()

