import gensim
from gensim.models import KeyedVectors
import numpy as np
from typing import List, Dict, Optional, Tuple, Any  # тип модели упростим до Any для совместимости с Pylance
from pathlib import Path
import json
from logger import get_logger

class MLClassifier:
    def __init__(self, models_dir: str = "models", use_pretrained: bool = True):
        self.logger = get_logger()
        # Модель может быть Word2Vec или KeyedVectors
        # Используем Any, чтобы избежать конфликтов с разными версиями gensim и статическим анализатором
        self.model: Optional[Any] = None
        self.is_trained = False
        self.category_vectors: Dict[str, np.ndarray] = {}
        self.models_dir = Path(models_dir)
        self.use_pretrained = use_pretrained
        self.is_pretrained = False  # Флаг для предобученной модели
        self._shape_warning_emitted = False
        
    def train_word2vec(self, training_data: Dict[str, List[str]]) -> bool:
        """Обучает Word2Vec модель на текстах категорий"""
        try:
            # Подготавливаем данные для обучения
            all_sentences = []
            for category_texts in training_data.values():
                for text in category_texts:
                    tokens = self._tokenize_text(text)
                    all_sentences.append(tokens)
            
            if not all_sentences:
                self.logger.warning("Нет данных для обучения Word2Vec")
                return False
            
            # Обучаем модель
            self.model = gensim.models.Word2Vec(
                sentences=all_sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                sg=1  # skip-gram
            )
            
            # Устанавливаем флаг ДО создания векторов категорий
            # чтобы text_to_vector мог работать
            self.is_trained = True
            
            # Создаем эталонные векторы для категорий
            self._create_category_vectors(training_data)
            self.logger.info("✅ Word2Vec модель обучена")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обучения Word2Vec: {e}")
            return False
    
    def _create_category_vectors(self, training_data: Dict[str, List[str]]):
        """Создает эталонные векторы для каждой категории"""
        self.logger.info("🔧 Создание векторов категорий...")
        
        for category, texts in training_data.items():
            self.logger.info(f"   Обрабатываем категорию '{category}': {len(texts)} текстов")
            category_vectors = []
            
            for i, text in enumerate(texts):
                text_vector = self.text_to_vector(text)
                if text_vector is not None:
                    category_vectors.append(text_vector)
                    self.logger.info(f"     Текст {i+1}: вектор создан (длина: {len(text_vector)})")
                else:
                    self.logger.warning(f"     Текст {i+1}: не удалось создать вектор")
            
            if category_vectors:
                self.category_vectors[category] = np.mean(category_vectors, axis=0)
                self.logger.info(f"   ✅ Вектор для '{category}' создан (длина: {len(self.category_vectors[category])})")
            else:
                self.logger.error(f"   ❌ Не удалось создать вектор для '{category}' - нет валидных текстов")
    
    def _get_word_vectors(self, word: str) -> Optional[np.ndarray]:
        """Получает вектор слова, учитывая тип модели (Word2Vec или KeyedVectors)"""
        # Если модель не загружена – вектора нет
        if self.model is None:
            return None
        
        if self.is_pretrained:
            # Предобученная модель (KeyedVectors)
            if hasattr(self.model, "key_to_index") and word in self.model.key_to_index:
                return self.model[word]  # type: ignore[index]
        else:
            # Обычная модель (Word2Vec)
            if hasattr(self.model, 'wv') and word in self.model.wv.key_to_index:
                return self.model.wv[word]  # type: ignore[union-attr]
        return None
    
    def _expected_vector_size(self) -> int:
        """Возвращает ожидаемую размерность векторов для текущей модели"""
        if self.model is None:
            return 0
        if self.is_pretrained:
            return getattr(self.model, 'vector_size', 0)
        if hasattr(self.model, 'wv'):
            return getattr(self.model.wv, 'vector_size', 0)
        return getattr(self.model, 'vector_size', 0)
    
    def _validate_category_vectors(self):
        """Проверяет, что сохраненные векторы категорий подходят текущей модели"""
        expected_size = self._expected_vector_size()
        if expected_size == 0 or not self.category_vectors:
            return
        
        invalid_categories = [
            category for category, vector in self.category_vectors.items()
            if len(vector) != expected_size
        ]
        
        if invalid_categories:
            for category in invalid_categories:
                self.logger.warning(
                    f"⚠️  Вектор категории '{category}' имеет размер {len(self.category_vectors[category])}, "
                    f"но модель ожидает {expected_size}. Вектор будет удален."
                )
                self.category_vectors.pop(category, None)
            
            if not self.category_vectors:
                self.logger.error(
                    "❌ Ни один вектор категории не соответствует размеру модели. "
                    "ML классификация будет отключена до пересоздания векторов."
                )
    def _find_word_in_vocab(self, word: str) -> Optional[str]:
        """Ищет слово в словаре, возвращает найденный вариант"""
        model = self.model
        if model is None:
            return None

        # Прямой поиск
        if self.is_pretrained:
            key_index = getattr(model, "key_to_index", {})
            if word in key_index:
                return word
        else:
            if hasattr(model, "wv"):
                key_index = getattr(model.wv, "key_to_index", {})
                if word in key_index:
                    return word

        # Набор морфологических тегов (используется только для предобученной модели)
        tags = ['_NOUN', '_VERB', '_ADJ', '_ADV', '_PRON', '_DET', '_PREP', '_CONJ']

        # Если предобученная модель с морфологическими тегами
        if self.is_pretrained:
            key_index = getattr(model, "key_to_index", {})
            # Пробуем варианты с тегами
            for tag in tags:
                word_with_tag = word + tag
                if word_with_tag in key_index:
                    return word_with_tag

        # Пытаемся найти базовые формы (упрощенная лемматизация)
        variants = [
            word,  # оригинал
            word.rstrip('уеыаоэяию'),  # без последней гласной
            word.rstrip('уеыаоэяию').rstrip('уеыаоэяию'),  # без двух последних гласных
        ]

        for variant in variants:
            if not variant:
                continue

            if self.is_pretrained:
                key_index = getattr(model, "key_to_index", {})
                if variant in key_index:
                    return variant
                # Пробуем с тегами
                for tag in tags:
                    variant_with_tag = variant + tag
                    if variant_with_tag in key_index:
                        return variant_with_tag
            else:
                if hasattr(model, "wv"):
                    key_index = getattr(model.wv, "key_to_index", {})
                    if variant in key_index:
                        return variant

        return None
    
    def text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """Преобразует текст в вектор"""
        if not self.is_trained or self.model is None or not text:
            self.logger.debug(f"text_to_vector: модель не готова или текст пустой (is_trained={self.is_trained}, model={self.model is not None}, text={bool(text)})")
            return None
        
        tokens = self._tokenize_text(text)
        self.logger.debug(f"Токены для векторизации: {tokens}")
        
        if not tokens:
            self.logger.debug("Нет токенов после токенизации")
            return None
        
        vectors = []
        
        for token in tokens:
            # Ищем слово в словаре (с учетом морфологических тегов)
            found_word = self._find_word_in_vocab(token)
            if found_word:
                vector = self._get_word_vectors(found_word)
                if vector is not None:
                    vectors.append(vector)
                    self.logger.debug(f"Токен '{token}' найден как '{found_word}'")
            else:
                self.logger.debug(f"Токен '{token}' НЕ найден в словаре")
        
        self.logger.debug(f"Найдено векторов: {len(vectors)} из {len(tokens)} токенов")
        
        if vectors:
            result = np.mean(vectors, axis=0)
            self.logger.debug(f"Вектор создан (длина: {len(result)})")
            return result
        else:
            self.logger.warning(f"Не удалось создать вектор - нет токенов в словаре. Токены: {tokens}")
            return None
    
    def predict_category(self, text: str) -> Tuple[Optional[str], float]:
        """Предсказывает категорию с уверенностью"""
        if not self.is_trained or not self.category_vectors:
            return None, 0.0
        
        text_vector = self.text_to_vector(text)
        if text_vector is None:
            return None, 0.0
        
        # Ищем ближайшую категорию по косинусной близости
        best_category = None
        best_similarity = -1.0
        
        for category, category_vector in self.category_vectors.items():
            similarity = self._cosine_similarity(text_vector, category_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        return best_category, best_similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусную близость между векторами"""
        if vec1.shape != vec2.shape:
            if not self._shape_warning_emitted:
                self.logger.warning(
                    f"⚠️  Несовместимые размеры векторов для cosine similarity: "
                    f"{vec1.shape} vs {vec2.shape}. Возвращаю 0."
                )
                self._shape_warning_emitted = True
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Токенизирует текст (исправленная версия)"""
        import re
        
        if not text:
            return []
        
        # Более мягкая очистка - только специальные символы
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Разбиваем на слова
        tokens = text_clean.split()
        
        # НЕ удаляем короткие слова - они могут быть важны!
        # tokens = [token for token in tokens if len(token) > 2]
        
        # Вместо этого убираем только совсем пустые токены
        tokens = [token for token in tokens if token.strip()]
        
        self.logger.debug(f"Токенизация: '{text[:50]}...' -> {len(tokens)} токенов")
        
        return tokens
    
    def save_model(self) -> bool:
        """Сохраняет модель в папку models/"""
        try:
            if self.model is None:
                self.logger.error("❌ Нечего сохранять: модель не обучена")
                return False
            
            # Предобученные модели (KeyedVectors) не нужно сохранять - они уже есть
            if self.is_pretrained:
                self.logger.info("💡 Предобученная модель уже сохранена, сохраняю только векторы категорий")
            else:
                self.models_dir.mkdir(parents=True, exist_ok=True)
                
                # Сохраняем Word2Vec модель
                model_path = self.models_dir / "word2vec.model"
                self.model.save(str(model_path))
                self.logger.info(f"💾 Word2Vec модель сохранена в {model_path}")
            
            # Сохраняем векторы категорий (ВАЖНО: даже если пустые)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            vectors_data = {
                category: vector.tolist() 
                for category, vector in self.category_vectors.items()
            }
            
            vectors_path = self.models_dir / "category_vectors.json"
            with open(vectors_path, 'w', encoding='utf-8') as f:
                json.dump(vectors_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Векторы категорий сохранены: {list(vectors_data.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False
    
    def load_pretrained_model(self, model_name: str = "word2vec-ruscorpora-300.model") -> bool:
        """Загружает предобученную модель (KeyedVectors)"""
        try:
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                self.logger.warning(f"Предобученная модель не найдена: {model_path}")
                return False
            
            # Пробуем загрузить как KeyedVectors (предобученная модель)
            try:
                self.model = KeyedVectors.load(str(model_path))
                self.is_pretrained = True
                model = self.model
                self.logger.info(f"📥 Предобученная модель загружена: {model_name}")
                if model is not None:
                    vocab_size = len(getattr(model, "key_to_index", {}))
                    vector_size = getattr(model, "vector_size", 0)
                    self.logger.info(f"   Размер словаря: {vocab_size} слов")
                    self.logger.info(f"   Размер вектора: {vector_size}")
            except Exception:
                # Если не KeyedVectors, пробуем как Word2Vec
                self.model = gensim.models.Word2Vec.load(str(model_path))
                self.is_pretrained = False
                self.logger.info(f"📥 Word2Vec модель загружена: {model_name}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки предобученной модели: {e}")
            return False
    
    def load_model(self) -> bool:
        """Загружает модель из папки models/ (сначала пробует предобученную, потом свою)"""
        try:
            vectors_path = self.models_dir / "category_vectors.json"
            
            # Сначала пробуем загрузить предобученную модель
            if self.use_pretrained:
                pretrained_path = self.models_dir / "word2vec-ruscorpora-300.model"
                if pretrained_path.exists():
                    if self.load_pretrained_model("word2vec-ruscorpora-300.model"):
                        # Загружаем векторы категорий (если есть)
                        if vectors_path.exists():
                            with open(vectors_path, 'r', encoding='utf-8') as f:
                                vectors_data = json.load(f)
                            
                            self.category_vectors = {
                                category: np.array(vector) 
                                for category, vector in vectors_data.items()
                            }
                            self._validate_category_vectors()
                            if self.category_vectors:
                                self.logger.info(f"📥 Векторы категорий загружены: {list(self.category_vectors.keys())}")
                            else:
                                self.logger.warning("Векторы категорий пустые или несовместимы - нужно создать")
                        else:
                            self.logger.warning("Векторы категорий не найдены - нужно создать")
                        return True
            
            # Если предобученная не найдена, пробуем свою модель
            model_path = self.models_dir / "word2vec.model"
            
            if not model_path.exists():
                self.logger.warning("Word2Vec модель не найдена")
                return False
            
            # Загружаем Word2Vec модель
            self.model = gensim.models.Word2Vec.load(str(model_path))
            self.is_pretrained = False
            self.logger.info(f"📥 Word2Vec модель загружена")
            
            # Загружаем векторы категорий (если есть)
            if vectors_path.exists():
                with open(vectors_path, 'r', encoding='utf-8') as f:
                    vectors_data = json.load(f)
                
                self.category_vectors = {
                    category: np.array(vector) 
                    for category, vector in vectors_data.items()
                }
                self._validate_category_vectors()
                if self.category_vectors:
                    self.logger.info(f"📥 Векторы категорий загружены: {list(self.category_vectors.keys())}")
                else:
                    self.logger.warning("Векторы категорий отсутствуют или несовместимы")
            else:
                self.logger.warning("Векторы категорий не найдены")
                self.category_vectors = {}
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:  # ✅ Используем Any вместо any
        """Возвращает информацию о модели"""
        if self.model is None:
            return {"status": "not_trained"}
        
        # Определяем размер словаря и вектора в зависимости от типа модели
        if self.is_pretrained:
            vocab_size = len(getattr(self.model, "key_to_index", {}))
            vector_size = getattr(self.model, "vector_size", 0)
            model_type = "pretrained (KeyedVectors)"
        else:
            if hasattr(self.model, "wv"):
                vocab_size = len(getattr(self.model.wv, "key_to_index", {}))
                vector_size = getattr(self.model.wv, "vector_size", 0)
            else:
                vocab_size = 0
                vector_size = getattr(self.model, "vector_size", 0)
            model_type = "trained (Word2Vec)"
        
        return {
            "status": "trained" if self.is_trained else "not_trained",
            "model_type": model_type,
            "vocabulary_size": vocab_size,
            "categories": list(self.category_vectors.keys()),
            "vector_size": vector_size
        }