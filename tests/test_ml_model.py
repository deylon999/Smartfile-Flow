"""
Тесты для ML модели
"""
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
import numpy as np
from ml_model import MLClassifier


class TestMLModel(unittest.TestCase):
    """Тесты для MLClassifier"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.classifier = MLClassifier(use_pretrained=False)
        self.training_data = {
            'work': [
                "работа проект задача",
                "проект разработка тестирование",
            ],
            'finance': [
                "деньги банк счет",
                "бюджет финансы оплата",
            ]
        }
    
    def test_train_model(self):
        """Тест обучения модели"""
        result = self.classifier.train_word2vec(self.training_data)
        self.assertTrue(result, "Модель должна обучиться успешно")
        self.assertTrue(self.classifier.is_trained, "Флаг is_trained должен быть True")
        self.assertIsNotNone(self.classifier.model, "Модель должна быть создана")
    
    def test_text_to_vector(self):
        """Тест преобразования текста в вектор"""
        # Обучаем модель
        self.classifier.train_word2vec(self.training_data)
        
        # Тестируем text_to_vector
        text = "работа проект"
        vector = self.classifier.text_to_vector(text)
        
        self.assertIsNotNone(vector, "Вектор должен быть создан")
        # После проверки на None явно приводим к np.ndarray для статического анализатора
        assert vector is not None
        self.assertEqual(len(vector), 100, "Размер вектора должен быть 100")
        self.assertIsInstance(vector, np.ndarray, "Вектор должен быть numpy array")
    
    def test_text_to_vector_empty_text(self):
        """Тест text_to_vector с пустым текстом"""
        self.classifier.train_word2vec(self.training_data)
        
        vector = self.classifier.text_to_vector("")
        self.assertIsNone(vector, "Пустой текст должен вернуть None")
    
    def test_text_to_vector_unknown_words(self):
        """Тест text_to_vector с неизвестными словами"""
        self.classifier.train_word2vec(self.training_data)
        
        # Слова, которых нет в обучающих данных
        vector = self.classifier.text_to_vector("абсолютно_неизвестные_слова_xyz")
        
        # Может вернуть None или вектор (в зависимости от обработки)
        # Проверяем, что не падает с ошибкой
        self.assertIsInstance(vector, (np.ndarray, type(None)), 
                             "Должен вернуть вектор или None")
    
    def test_create_category_vectors(self):
        """Тест создания векторов категорий"""
        self.classifier.train_word2vec(self.training_data)
        
        self.assertGreater(len(self.classifier.category_vectors), 0, 
                          "Векторы категорий должны быть созданы")
        
        # Проверяем, что все категории имеют векторы
        for category in self.training_data.keys():
            self.assertIn(category, self.classifier.category_vectors,
                         f"Категория {category} должна иметь вектор")
            vector = self.classifier.category_vectors[category]
            self.assertEqual(len(vector), 100, 
                           f"Вектор категории {category} должен быть размером 100")
    
    def test_predict_category(self):
        """Тест предсказания категории"""
        self.classifier.train_word2vec(self.training_data)
        
        # Тестируем предсказание
        category, confidence = self.classifier.predict_category("работа проект задача")
        
        self.assertIsNotNone(category, "Категория должна быть предсказана")
        self.assertIn(category, self.training_data.keys(), 
                     "Категория должна быть из обучающих данных")
        self.assertGreaterEqual(confidence, 0.0, "Уверенность должна быть >= 0")
        self.assertLessEqual(confidence, 1.0, "Уверенность должна быть <= 1")
    
    def test_predict_category_empty_vectors(self):
        """Тест предсказания без векторов категорий"""
        # Модель не обучена
        category, confidence = self.classifier.predict_category("работа")
        
        self.assertIsNone(category, "Без обучения категория должна быть None")
        self.assertEqual(confidence, 0.0, "Уверенность должна быть 0")
    
    def test_save_and_load_model(self):
        """Тест сохранения и загрузки модели"""
        # Обучаем и сохраняем
        self.classifier.train_word2vec(self.training_data)
        self.classifier.save_model()
        
        # Создаем новый классификатор и загружаем
        new_classifier = MLClassifier(use_pretrained=False)
        result = new_classifier.load_model()
        
        self.assertTrue(result, "Модель должна загрузиться")
        self.assertTrue(new_classifier.is_trained, "Флаг is_trained должен быть True")
        self.assertGreater(len(new_classifier.category_vectors), 0,
                          "Векторы категорий должны загрузиться")
    
    def test_get_model_info(self):
        """Тест получения информации о модели"""
        # Без модели
        info = self.classifier.get_model_info()
        self.assertEqual(info['status'], 'not_trained')
        
        # С моделью
        self.classifier.train_word2vec(self.training_data)
        info = self.classifier.get_model_info()
        
        self.assertEqual(info['status'], 'trained')
        self.assertGreater(info['vocabulary_size'], 0)
        self.assertEqual(info['vector_size'], 100)
        self.assertIn('categories', info)
    
    def test_cosine_similarity_dimension_mismatch(self):
        """Проверяем, что косинус не падает при разной длине векторов"""
        vec1 = np.zeros(100)
        vec2 = np.zeros(50)
        result = self.classifier._cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)
    
    def test_validate_category_vectors_mismatch(self):
        """Несовместимые векторы категорий должны удаляться"""
        class DummyModel:
            def __init__(self, vector_size):
                self.vector_size = vector_size
        self.classifier.model = DummyModel(vector_size=300)
        self.classifier.is_pretrained = True
        self.classifier.category_vectors = {
            'work': np.zeros(100),
            'finance': np.zeros(300)
        }
        self.classifier._validate_category_vectors()
        self.assertIn('finance', self.classifier.category_vectors)
        self.assertNotIn('work', self.classifier.category_vectors)


if __name__ == '__main__':
    unittest.main()

