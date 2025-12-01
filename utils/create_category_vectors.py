"""
Скрипт для создания векторов категорий из файлов в data/sorted/
Использует предобученную модель для векторизации
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_sorter import FileSorter
from ml_model import MLClassifier
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def collect_texts_from_sorted_files():
    """Собирает тексты из файлов в data/sorted/ по категориям"""
    print("=" * 70)
    print("📚 СБОР ДАТАСЕТА ИЗ ФАЙЛОВ")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    sorted_dir = project_root / 'data' / 'sorted'
    
    if not sorted_dir.exists():
        print(f"❌ Папка {sorted_dir} не найдена!")
        return {}
    
    # Создаем FileSorter для извлечения текстов
    sorter = FileSorter("data/raw", "data/sorted")
    
    # Маппинг папок к категориям
    category_mapping = {
        'work': 'work',
        'finance': 'finance',
        'personal': 'personal',
        'study': 'study',
        'other': 'other'
    }
    
    training_data = {}
    
    print("\n📂 Сканируем файлы по категориям...")
    
    for folder_name, category in category_mapping.items():
        category_dir = sorted_dir / folder_name
        
        if not category_dir.exists():
            print(f"   ⚠️  Папка {folder_name} не найдена, пропускаем")
            continue
        
        texts = []
        files = list(category_dir.glob('*'))
        
        print(f"\n   📁 Категория '{category}' ({folder_name}/):")
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            print(f"      Читаю: {file_path.name}...", end=' ')
            
            text = sorter.extract_text_from_file(file_path)
            
            if text and text.strip():
                # Ограничиваем длину текста (берем первые 5000 символов)
                text = text.strip()[:5000]
                texts.append(text)
                print(f"✅ ({len(text)} символов)")
            else:
                print("❌ не удалось извлечь текст")
        
        if texts:
            training_data[category] = texts
            print(f"   ✅ Собрано {len(texts)} текстов для категории '{category}'")
        else:
            print(f"   ⚠️  Нет текстов для категории '{category}'")
    
    print(f"\n📊 ИТОГО:")
    total_texts = sum(len(texts) for texts in training_data.values())
    print(f"   Категорий: {len(training_data)}")
    print(f"   Всего текстов: {total_texts}")
    for category, texts in training_data.items():
        print(f"   - {category}: {len(texts)} текстов")
    
    return training_data

def create_category_vectors(training_data):
    """Создает векторы категорий используя предобученную модель"""
    print("\n" + "=" * 70)
    print("🔧 СОЗДАНИЕ ВЕКТОРОВ КАТЕГОРИЙ")
    print("=" * 70)
    
    if not training_data:
        print("❌ Нет данных для создания векторов!")
        return False
    
    # Создаем классификатор с предобученной моделью
    print("\n1️⃣  Загружаем предобученную модель...")
    classifier = MLClassifier(use_pretrained=True)
    
    if not classifier.load_model():
        print("❌ Не удалось загрузить предобученную модель!")
        return False
    
    print("✅ Модель загружена!")
    
    # Создаем векторы категорий
    print("\n2️⃣  Создаем векторы категорий...")
    classifier._create_category_vectors(training_data)
    
    if not classifier.category_vectors:
        print("❌ Не удалось создать векторы категорий!")
        return False
    
    print(f"\n✅ Векторы категорий созданы:")
    for category, vector in classifier.category_vectors.items():
        print(f"   - {category}: размер {len(vector)}")
    
    # Сохраняем векторы
    print("\n3️⃣  Сохраняем векторы категорий...")
    if classifier.save_model():
        print("✅ Векторы категорий сохранены!")
        return True
    else:
        print("❌ Ошибка при сохранении векторов!")
        return False

def main():
    """Основная функция"""
    # Шаг 1: Собираем тексты из файлов
    training_data = collect_texts_from_sorted_files()
    
    if not training_data:
        print("\n❌ Не удалось собрать данные из файлов!")
        print("💡 Убедитесь, что файлы находятся в data/sorted/ по категориям")
        return
    
    # Шаг 2: Создаем векторы категорий
    success = create_category_vectors(training_data)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ ВСЕ ГОТОВО!")
        print("=" * 70)
        print("\n💡 Следующие шаги:")
        print("   1. Включите ML в config.yaml: use_ml: true")
        print("   2. Протестируйте сортировку файлов")
    else:
        print("\n❌ Ошибка при создании векторов категорий!")

if __name__ == '__main__':
    main()

