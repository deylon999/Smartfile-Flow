"""
Главный файл для запуска умного сортировщика файлов
"""
import sys
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from file_sorter import FileSorter
from logger import get_logger

def main():
    """Главная функция с CLI интерфейсом"""
    parser = argparse.ArgumentParser(
        description='Умный сортировщик файлов с ML классификацией',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                                    # Сортировка из data/raw в data/sorted
  python main.py --source ./documents --target ./sorted
  python main.py --source ./files --copy           # Копировать вместо перемещения
  python main.py --source ./files --no-ml           # Использовать только правила
  python main.py --conflict skip                   # Пропускать дубликаты
  python main.py --conflict overwrite              # Перезаписывать дубликаты
  python main.py --conflict rename                 # Переименовывать дубликаты
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='data/raw',
        help='Папка с исходными файлами (по умолчанию: data/raw)'
    )
    
    parser.add_argument(
        '--target', '-t',
        type=str,
        default='data/sorted',
        help='Папка для отсортированных файлов (по умолчанию: data/sorted)'
    )
    
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Копировать файлы вместо перемещения'
    )
    
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Отключить ML классификацию, использовать только правила'
    )
    
    parser.add_argument(
        '--conflict',
        choices=['skip', 'overwrite', 'rename'],
        help='Стратегия обработки конфликтов: skip (пропустить), overwrite (перезаписать), rename (переименовать)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Отключить прогресс-бар'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    logger = get_logger()
    if args.verbose:
        import logging
        logging.getLogger('FileSorter').setLevel(logging.DEBUG)
    
    # Проверяем существование исходной папки
    source_dir = Path(args.source)
    if not source_dir.exists():
        logger.error(f"❌ Исходная папка не найдена: {source_dir}")
        logger.info(f"💡 Создайте папку или укажите правильный путь")
        return 1
    
    # Обновляем конфиг если нужно
    if args.copy:
        from config import get_config
        config = get_config()
        config.settings.copy_files = True
        logger.info("📋 Режим: копирование файлов")
    else:
        logger.info("📋 Режим: перемещение файлов")
    
    if args.no_ml:
        from config import get_config
        config = get_config()
        config.settings.use_ml = False
        logger.info("📋 ML классификация отключена, используются только правила")
    
    if args.conflict:
        from config import get_config
        config = get_config()
        config.settings.conflict_resolution = args.conflict
        logger.info(f"📋 Стратегия конфликтов: {args.conflict}")
    
    # Создаем сортировщик
    try:
        sorter = FileSorter(str(source_dir), args.target)
        
        # Запускаем сортировку
        logger.info("🚀 Начинаем сортировку файлов...")
        show_progress = not args.no_progress
        sorter.sort_all(show_progress=show_progress)
        
        logger.info("✅ Сортировка завершена!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Сортировка прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

