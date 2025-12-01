import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from logger import get_logger
from config import get_config
from ml_model import MLClassifier

class FileSorter:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.logger = get_logger()
        self.config = get_config()
        # Используем предобученную модель по умолчанию
        self.ml_classifier = MLClassifier(use_pretrained=True)
        
        # Пытаемся загрузить обученную ML модель
        self._load_ml_model()
        
        self._create_category_folders()
    
    def _load_ml_model(self):
        """Загружает ML модель если она есть и включена в конфиге"""
        if self.config.settings.use_ml:
            if not self.ml_classifier.load_model():
                self.logger.warning("ML модель не найдена. Будет использоваться rule-based подход")
    
    def _create_category_folders(self):
        """Создает папки для категорий из конфига"""
        self.target_dir.mkdir(parents=True, exist_ok=True)
        for category_name in self.config.get_category_names():
            category_path = self.target_dir / category_name
            category_path.mkdir(parents=True, exist_ok=True)
    
    def _read_text_with_encoding(self, file_path: Path) -> Optional[str]:
        """Читает текстовый файл, определяя кодировку"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read()
        except FileNotFoundError:
            self.logger.error(f"Файл не найден: {file_path}")
            return None
        except OSError as exc:
            self.logger.error(f"Ошибка чтения {file_path}: {exc}")
            return None
        
        if not raw_data:
            return ""
        
        encoding = chardet.detect(raw_data).get('encoding') or 'utf-8'
        try:
            return raw_data.decode(encoding, errors='ignore')
        except LookupError:
            self.logger.warning(f"Неизвестная кодировка '{encoding}' для {file_path}, используется UTF-8")
            return raw_data.decode('utf-8', errors='ignore')
    
    def _collect_json_text(self, data: Any, collector: List[str], depth: int = 0, max_depth: int = 32):
        """Рекурсивно собирает текст из JSON, ограничивая глубину"""
        if depth > max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                collector.append(str(key))
                self._collect_json_text(value, collector, depth + 1, max_depth)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                self._collect_json_text(item, collector, depth + 1, max_depth)
        elif data is None:
            return
        else:
            collector.append(str(data))
    
    def _extract_text_from_json(self, file_path: Path) -> Optional[str]:
        """Извлекает текст из JSON файла"""
        decoded_text = self._read_text_with_encoding(file_path)
        if decoded_text is None:
            return None
        if not decoded_text.strip():
            return ""
        
        try:
            payload = json.loads(decoded_text)
        except json.JSONDecodeError as exc:
            self.logger.error(f"Ошибка разбора JSON {file_path}: {exc}")
            return None
        
        collected: List[str] = []
        self._collect_json_text(payload, collected)
        return " ".join(collected).strip()
    
    def _extract_text_from_xml(self, file_path: Path) -> Optional[str]:
        """Извлекает текст из XML файла с безопасным парсером"""
        try:
            from defusedxml import ElementTree as ET
            parser_name = "defusedxml"
        except ImportError:
            import xml.etree.ElementTree as ET
            parser_name = "xml.etree"
            self.logger.warning("defusedxml не установлен, используется стандартный XML-парсер")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except (ET.ParseError, OSError) as exc:
            self.logger.error(f"Ошибка разбора XML {file_path}: {exc}")
            return None
        
        if root is None:
            self.logger.debug(f"XML файл {file_path} ({parser_name}): корневой элемент отсутствует")
            return ""
        
        fragments: List[str] = []
        for elem in list(root.iter()):
            if elem.text and elem.text.strip():
                fragments.append(elem.text.strip())
            for attr_val in elem.attrib.values():
                if attr_val:
                    fragments.append(str(attr_val))
            if elem.tail and elem.tail.strip():
                fragments.append(elem.tail.strip())
        
        if not fragments:
            self.logger.debug(f"XML файл {file_path} ({parser_name}) не содержит текста")
        return " ".join(fragments).strip()
    
    def extract_text_from_file(self, file_path: Path) -> Optional[str]:
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.txt':
                return self._read_text_with_encoding(file_path)
                    
            elif file_ext == '.pdf':
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + " "
                    return text.strip()
                except ImportError:
                    self.logger.warning("pdfplumber не установлен. Используется PyPDF2")
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])
                    return text
                    
            elif file_ext in ['.docx', '.doc']:
                import docx
                doc = docx.Document(str(file_path))
                text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
                return text
            
            elif file_ext == '.json':
                return self._extract_text_from_json(file_path)
            
            elif file_ext == '.xml':
                return self._extract_text_from_xml(file_path)
                
            else:
                self.logger.warning(f"Неподдерживаемый формат файла: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка при чтении файла {file_path}: {e}")
            return None
    
    def categorize_with_rules(self, text: str) -> Tuple[str, float]:
        """Категоризация на основе правил (взвешенные ключевые слова)"""
        if not text:
            return 'other', 0.0
        
        text_lower = text.lower()
        category_scores: Dict[str, float] = {}
        
        for category_name in self.config.get_category_names():
            if category_name == 'other':
                continue
                
            weighted_keywords = self.config.get_weighted_keywords(category_name)
            score = 0.0
            
            for keyword, weight in weighted_keywords:
                count = text_lower.count(keyword)
                score += count * weight
                
                # Бонус за множественные вхождения
                if count > 1:
                    score += count * 0.2
            
            if score >= self.config.settings.min_confidence_score:
                category_scores[category_name] = score
        
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            confidence = category_scores[best_category]
            return best_category, confidence
        
        return 'other', 0.0
    
    def categorize_with_ml(self, text: str) -> Tuple[str, float]:
        """Категоризация с использованием ML"""
        if not self.config.settings.use_ml or not self.ml_classifier.is_trained:
            return self.categorize_with_rules(text)
        
        category, confidence = self.ml_classifier.predict_category(text)
        
        if category and confidence >= self.config.settings.ml_confidence_threshold:
            return category, confidence
        else:
            # Fallback на правила если ML не уверен
            return self.categorize_with_rules(text)
    
    def categorize_file(self, text: str) -> str:
        """Основной метод категоризации"""
        if self.config.settings.use_ml and self.ml_classifier.is_trained:
            category, confidence = self.categorize_with_ml(text)
            method = "ML"
            # Для ML: косинусная близость от 0 до 1
            if confidence > 0.8:
                self.logger.info(f"🎯 {category}: высокая уверенность {confidence:.2f} ({method})")
            elif confidence > 0.5:
                self.logger.info(f"✅ {category}: уверенность {confidence:.2f} ({method})")
            elif confidence >= self.config.settings.ml_confidence_threshold:
                self.logger.info(f"🤔 {category}: низкая уверенность {confidence:.2f} ({method})")
            else:
                self.logger.info(f"⚠️ {category}: очень низкая уверенность {confidence:.2f} ({method})")
        else:
            category, confidence = self.categorize_with_rules(text)
            method = "правила"
            # Для правил: score может быть больше 1.0
            if confidence > 5.0:
                self.logger.info(f"🎯 {category}: высокая уверенность {confidence:.1f} ({method})")
            elif confidence > 2.0:
                self.logger.info(f"✅ {category}: уверенность {confidence:.1f} ({method})")
            else:
                self.logger.info(f"🤔 {category}: низкая уверенность {confidence:.1f} ({method})")
        
        return category
    
    def scan_directory(self) -> List[Path]:
        files = []
        
        if not self.source_dir.exists():
            self.logger.error(f"Директория {self.source_dir} не существует!")
            return files
        
        for ext in self.config.settings.supported_extensions:
            for file_path in self.source_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    files.append(file_path)
        
        return files
    
    def _log_with_tqdm(self, message: str, level: str = 'info'):
        """Логирует сообщение с поддержкой tqdm"""
        try:
            from tqdm import tqdm
            instances = getattr(tqdm, "_instances", None)
            if instances is not None and len(instances) > 0:
                tqdm.write(message)
                # Также логируем в обычный логгер для файла
                if level == 'warning':
                    self.logger.warning(message)
                elif level == 'error':
                    self.logger.error(message)
                else:
                    self.logger.info(message)
            else:
                if level == 'warning':
                    self.logger.warning(message)
                elif level == 'error':
                    self.logger.error(message)
                else:
                    self.logger.info(message)
        except (ImportError, AttributeError):
            if level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            else:
                self.logger.info(message)
    
    def _resolve_conflict(self, target_path: Path) -> Tuple[Path, bool]:
        """
        Разрешает конфликт имен файлов согласно настройке conflict_resolution
        Возвращает: (путь к файлу, нужно ли обрабатывать файл)
        """
        if not target_path.exists():
            return target_path, True
        
        resolution = self.config.settings.conflict_resolution
        
        if resolution == 'skip':
            # Пропускаем файл
            message = f"⏭️  Файл {target_path.name} уже существует, пропускаем (conflict_resolution=skip)"
            self._log_with_tqdm(message, level='warning')
            return target_path, False
        
        elif resolution == 'overwrite':
            # Перезаписываем
            message = f"⚠️  Файл {target_path.name} уже существует, перезаписываем (conflict_resolution=overwrite)"
            self._log_with_tqdm(message, level='warning')
            return target_path, True
        
        elif resolution == 'rename':
            # Переименовываем
            stem = target_path.stem
            suffix = target_path.suffix
            parent = target_path.parent
            
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                new_path = parent / new_name
                if not new_path.exists():
                    message = f"📝 Файл переименован: {target_path.name} -> {new_name} (дубликат)"
                    self._log_with_tqdm(message, level='info')
                    return new_path, True
                counter += 1
        
        else:
            # Неизвестная стратегия, используем rename по умолчанию
            self.logger.warning(f"⚠️  Неизвестная стратегия конфликта '{resolution}', используем 'rename'")
            return self._resolve_conflict(target_path)  # Рекурсивно с rename
    
    def sort_file(self, file_path: Path) -> Optional[str]:
        """
        Сортирует файл
        Возвращает: 'sorted' - успешно, 'skipped' - пропущен, None - ошибка
        """
        text = self.extract_text_from_file(file_path)
        
        # Если не удалось извлечь текст, используем fallback
        if text is None or not text.strip():
            self._log_with_tqdm(f"⚠️  Не удалось извлечь текст из {file_path.name}, используем правила", level='warning')
            category = 'other'
        else:
            category = self.categorize_file(text)
        
        target_path = self.target_dir / category / file_path.name
        
        # Разрешаем конфликт согласно настройке
        target_path, should_process = self._resolve_conflict(target_path)
        
        if not should_process:
            return 'skipped'  # Файл пропущен
        
        try:
            if self.config.settings.copy_files:
                shutil.copy2(file_path, target_path)
                action = "скопирован"
            else:
                shutil.move(file_path, target_path)
                action = "перемещен"
            
            color = self.config.get_category_color(category)
            self.logger.info(f"{color} {file_path.name} -> {category}/ ({action})")
            return 'sorted'
            
        except Exception as e:
            error_msg = f"❌ Ошибка при обработке {file_path}: {e}"
            self._log_with_tqdm(error_msg, level='error')
            # Для ошибок всегда используем logger.error
            self.logger.error(error_msg)
            return None
    
    def sort_all(self, show_progress: bool = True):
        """
        Сортирует все файлы и возвращает статистику
        
        Args:
            show_progress: Показывать ли прогресс-бар (по умолчанию True)
        """
        self.logger.start_session(str(self.source_dir), str(self.target_dir))
        
        files = self.scan_directory()
        
        if not files:
            self.logger.warning("Файлы не найдены!")
            return {
                'total': 0,
                'sorted': 0,
                'failed': 0,
                'skipped': 0,
                'by_category': {},
                'method_used': 'none',
                'conflict_resolution': self.config.settings.conflict_resolution
            }
        
        self.logger.info(f"Найдено файлов: {len(files)}")
        
        sorted_count = 0
        failed_count = 0
        skipped_count = 0
        by_category = {}
        ml_used = 0
        rules_used = 0
        
        # Настраиваем прогресс-бар
        use_progress_bar = False
        try:
            if show_progress and len(files) > 1:
                from tqdm import tqdm
                # Используем tqdm.write для логирования, чтобы не конфликтовать с прогресс-баром
                file_iterator = tqdm(files, desc="📁 Обработка", unit="файл", 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]',
                                    ncols=100, mininterval=0.5)
                use_progress_bar = True
            else:
                file_iterator = files
        except ImportError:
            # Если tqdm не установлен, используем обычный итератор
            self.logger.debug("tqdm не установлен, прогресс-бар отключен")
            file_iterator = files
            use_progress_bar = False
        
        # Явно говорим типизатору, что когда use_progress_bar=True, у нас есть tqdm-объект
        progress_bar = None
        try:
            from tqdm import tqdm  # type: ignore
            if use_progress_bar and isinstance(file_iterator, tqdm):
                progress_bar = file_iterator  # type: ignore[assignment]
        except ImportError:
            progress_bar = None

        for file_path in file_iterator:
            # Определяем категорию для статистики (до перемещения)
            text = self.extract_text_from_file(file_path)
            if text:
                category = self.categorize_file(text)
                # Определяем метод
                if self.config.settings.use_ml and self.ml_classifier.is_trained:
                    ml_used += 1
                else:
                    rules_used += 1
            else:
                category = 'other'
                rules_used += 1
            
            result = self.sort_file(file_path)
            if result == 'sorted':
                sorted_count += 1
                by_category[category] = by_category.get(category, 0) + 1
            elif result == 'skipped':
                skipped_count += 1
            else:
                failed_count += 1
            
            # Обновляем прогресс-бар
            if progress_bar is not None:
                progress_bar.set_postfix({
                    '✅': sorted_count,
                    '⏭️': skipped_count,
                    '❌': failed_count
                })
        
        # Детальная статистика
        stats = {
            'total': len(files),
            'sorted': sorted_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'by_category': by_category,
            'method_used': 'ML' if ml_used > rules_used else 'rules' if rules_used > 0 else 'none',
            'ml_count': ml_used,
            'rules_count': rules_used,
            'conflict_resolution': self.config.settings.conflict_resolution
        }
        
        # Выводим статистику
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📊 СТАТИСТИКА СОРТИРОВКИ")
        self.logger.info("=" * 50)
        self.logger.info(f"Всего файлов: {stats['total']}")
        self.logger.info(f"Успешно отсортировано: {stats['sorted']}")
        if stats['skipped'] > 0:
            self.logger.info(f"Пропущено (дубликаты): {stats['skipped']}")
        if stats['failed'] > 0:
            self.logger.info(f"Ошибок: {stats['failed']}")
        self.logger.info(f"Метод: {stats['method_used']} (ML: {stats['ml_count']}, правила: {stats['rules_count']})")
        self.logger.info(f"Стратегия конфликтов: {stats['conflict_resolution']}")
        if stats['by_category']:
            self.logger.info("\nПо категориям:")
            for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
                color = self.config.get_category_color(category)
                self.logger.info(f"  {color} {category}: {count} файлов")
        self.logger.info("=" * 50)
        
        self.logger.end_session(sorted_count, len(files))
        
        return stats
    
    def train_ml_model(self, training_data: Dict[str, List[str]]) -> bool:
        """Обучает ML модель на предоставленных данных"""
        self.logger.info("🧠 Начинаем обучение ML модели...")
        
        success = self.ml_classifier.train_word2vec(training_data)
        
        if success:
            # Сохраняем обученную модель
            self.ml_classifier.save_model()
            self.logger.info("✅ ML модель обучена и сохранена")
        else:
            self.logger.error("❌ Не удалось обучить ML модель")
        
        return success

def main():
    project_root = Path(__file__).parent.parent
    source_dir = project_root / 'data' / 'raw'
    target_dir = project_root / 'data' / 'sorted'
    
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Тестовые файлы
    test_files = [
        ("работа.txt", "Это мой рабочий проект и задачи на неделю. Встреча с клиентом в пятницу."),
        ("финансы.txt", "Бюджет семьи на месяц, оплата счетов за банк и зарплата."),
        ("отпуск.txt", "Планы на отпуск с семьей и друзьями. Праздник в июле."),
        ("учеба.txt", "Лекции по курсу машинного обучения, домашнее задание к экзамену.")
    ]
    
    for filename, content in test_files:
        file_path = source_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    sorter = FileSorter(str(source_dir), str(target_dir))
    sorter.sort_all()

if __name__ == '__main__':
    main()