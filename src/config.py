import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from logger import get_logger

@dataclass
class CategoryConfig:
    name: str
    keywords: List[Tuple[str, float]]  # Теперь с весами
    color: str
    description: str

@dataclass
class SettingsConfig:
    supported_extensions: List[str]
    min_confidence_score: float  # Изменили на float
    log_retention_days: int
    copy_files: bool
    use_ml: bool
    ml_confidence_threshold: float
    conflict_resolution: str  # 'skip', 'overwrite', 'rename'

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_logger()
        self.config_path = Path(config_path)
        self.categories: Dict[str, CategoryConfig] = {}
        self.settings: SettingsConfig = self._create_default_settings()
        self._load_config()
    
    def _create_default_settings(self) -> SettingsConfig:
        return SettingsConfig(
            supported_extensions=['.txt', '.pdf', '.docx', '.doc', '.json', '.xml'],
            min_confidence_score=1.0,
            log_retention_days=30,
            copy_files=True,
            use_ml=False,
            ml_confidence_threshold=0.7,
            conflict_resolution='rename'  # По умолчанию переименовываем
        )
    
    def _load_config(self):
        if not self.config_path.exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return
            
            # Загружаем категории с весами
            if 'categories' in config_data:
                for category_name, category_data in config_data['categories'].items():
                    try:
                        # Валидация данных категории
                        if not isinstance(category_data, dict):
                            self.logger.warning(f"⚠️  Неверный формат категории '{category_name}', пропускаем")
                            continue
                        
                        # Валидация ключевых слов
                        keywords = category_data.get('keywords', [])
                        if not isinstance(keywords, list):
                            self.logger.warning(f"⚠️  Ключевые слова для '{category_name}' должны быть списком")
                            keywords = []
                        
                        # Конвертируем ключевые слова в формат (слово, вес)
                        weighted_keywords = []
                        for keyword_item in keywords:
                            try:
                                if isinstance(keyword_item, list) and len(keyword_item) == 2:
                                    weighted_keywords.append((str(keyword_item[0]), float(keyword_item[1])))
                                else:
                                    # Дефолтный вес 1.0 для обратной совместимости
                                    weighted_keywords.append((str(keyword_item), 1.0))
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"⚠️  Пропускаем неверное ключевое слово в '{category_name}': {e}")
                                continue
                        
                        # Валидация цвета и описания
                        color = category_data.get('color', '⚪')
                        description = category_data.get('description', '')
                        
                        self.categories[category_name] = CategoryConfig(
                            name=category_name,
                            keywords=weighted_keywords,
                            color=str(color),
                            description=str(description)
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка при загрузке категории '{category_name}': {e}")
                        continue
            
            # Загружаем настройки с валидацией
            if 'settings' in config_data:
                settings_data = config_data['settings']
                if not isinstance(settings_data, dict):
                    self.logger.warning("⚠️  Настройки должны быть словарем, используются значения по умолчанию")
                else:
                    try:
                        # Валидация расширений
                        extensions = settings_data.get('supported_extensions', self.settings.supported_extensions)
                        if not isinstance(extensions, list):
                            self.logger.warning("⚠️  supported_extensions должен быть списком")
                            extensions = self.settings.supported_extensions
                        
                        # Валидация числовых значений
                        try:
                            min_confidence = float(settings_data.get('min_confidence_score', self.settings.min_confidence_score))
                            if min_confidence < 0:
                                raise ValueError("min_confidence_score не может быть отрицательным")
                        except (ValueError, TypeError):
                            self.logger.warning("⚠️  Неверное значение min_confidence_score, используется значение по умолчанию")
                            min_confidence = self.settings.min_confidence_score
                        
                        try:
                            log_retention = int(settings_data.get('log_retention_days', self.settings.log_retention_days))
                            if log_retention < 0:
                                raise ValueError("log_retention_days не может быть отрицательным")
                        except (ValueError, TypeError):
                            self.logger.warning("⚠️  Неверное значение log_retention_days, используется значение по умолчанию")
                            log_retention = self.settings.log_retention_days
                        
                        try:
                            ml_threshold = float(settings_data.get('ml_confidence_threshold', self.settings.ml_confidence_threshold))
                            if not 0 <= ml_threshold <= 1:
                                raise ValueError("ml_confidence_threshold должен быть от 0 до 1")
                        except (ValueError, TypeError):
                            self.logger.warning("⚠️  Неверное значение ml_confidence_threshold, используется значение по умолчанию")
                            ml_threshold = self.settings.ml_confidence_threshold
                        
                        # Валидация conflict_resolution
                        conflict_resolution = settings_data.get('conflict_resolution', self.settings.conflict_resolution)
                        if conflict_resolution not in ['skip', 'overwrite', 'rename']:
                            self.logger.warning(f"⚠️  Неверное значение conflict_resolution '{conflict_resolution}', используется 'rename'")
                            conflict_resolution = 'rename'
                        
                        self.settings = SettingsConfig(
                            supported_extensions=extensions,
                            min_confidence_score=min_confidence,
                            log_retention_days=log_retention,
                            copy_files=bool(settings_data.get('copy_files', self.settings.copy_files)),
                            use_ml=bool(settings_data.get('use_ml', self.settings.use_ml)),
                            ml_confidence_threshold=ml_threshold,
                            conflict_resolution=conflict_resolution
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка при загрузке настроек: {e}, используются значения по умолчанию")
            
            self.logger.info(f"✅ Конфигурация загружена из {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
    
    def _create_default_config(self):
        default_config = {
            'categories': {
                'work': {
                    'keywords': [['работа', 2.0], ['проект', 1.5], ['задача', 1.5]],
                    'color': '🔵',
                    'description': 'Рабочие документы и проекты'
                },
                # ... остальные категории
            },
            'settings': {
                'supported_extensions': ['.txt', '.pdf', '.docx', '.doc', '.json', '.xml'],
                'min_confidence_score': 1.0,
                'log_retention_days': 30,
                'copy_files': True,
                'use_ml': False,
                'ml_confidence_threshold': 0.7
            }
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
            self.logger.info(f"📝 Создан конфигурационный файл: {self.config_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания конфигурационного файла: {e}")
    
    def get_category_names(self) -> List[str]:
        return list(self.categories.keys())
    
    def get_weighted_keywords(self, category_name: str) -> List[Tuple[str, float]]:
        return self.categories.get(category_name, CategoryConfig("", [], "", "")).keywords
    
    def get_category_color(self, category_name: str) -> str:
        return self.categories.get(category_name, CategoryConfig("", [], "", "")).color

_config_instance: Optional[Config] = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance