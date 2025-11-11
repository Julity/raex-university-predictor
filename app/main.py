# app/main.py
import streamlit as st
import pandas as pd
import sys
import os
import io
import numpy as np

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import logging
try:
    # Пытаемся определить, где мы запущены
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Вариант 1: Мы в папке app/ (Streamlit Cloud)
    if os.path.basename(current_dir) == 'app':
        src_path = os.path.join(current_dir, '..', 'src')
        models_path = os.path.join(current_dir, '..', 'models')
    # Вариант 2: Мы в корне проекта (локальная разработка)  
    else:
        src_path = os.path.join(current_dir, 'src')
        models_path = os.path.join(current_dir, 'models')
    
    # Добавляем пути в систему
    sys.path.insert(0, os.path.abspath(src_path))
    
    # Проверяем существование путей
    if not os.path.exists(models_path):
        logging.warning(f"Папка models не найдена: {models_path}")
    
except Exception as e:
    logging.error(f"Ошибка настройки путей: {e}")

# Теперь импортируем наши модули
try:
    from config import feature_order, russian_name
    from predictor import RAPredictor
except ImportError as e:
    logging.error(f"Ошибка импорта: {e}")
    st.error(f"Ошибка загрузки модулей: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="🎓 RAEX Rank Predictor", layout="wide")
st.title("🎓 RAEX Rank Predictor - Универсальная модель")

# Функция для безопасного преобразования значений
def safe_convert(value, default=0):
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

# Инициализация предсказателя
@st.cache_resource
def load_predictor():
    try:
        return RAPredictor()
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

predictor = load_predictor()

# Функция для проверки и обработки CSV файла
def process_csv_file(uploaded_file):
    try:
        # Пытаемся прочитать файл с разными кодировками
        for encoding in ['utf-8', 'cp1251', 'windows-1251']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error("Не удалось прочитать файл. Проверьте кодировку (должна быть UTF-8 или Windows-1251)")
            return None
        
        # Проверяем наличие необходимых колонок
        missing_features = set(feature_order) - set(df.columns)
        if missing_features:
            st.error(f"В файле отсутствуют следующие признаки: {missing_features}")
            st.info("Убедитесь, что файл содержит все необходимые колонки")
            return None
        
        # Заполняем пропущенные значения
        df = df.fillna(0)
        
        # Выбираем первую строку (первый вуз) для предсказания
        sample_data = {}
        for feat in feature_order:
            sample_data[feat] = safe_convert(df.iloc[0][feat])
        
        st.success(f"✅ Файл успешно загружен! Записей: {len(df)}")
        st.info(f"📝 Используются данные первого вуза из файла")
        
        return sample_data, df
    
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
        return None

# Инициализация session_state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = {}
if 'use_csv' not in st.session_state:
    st.session_state.use_csv = False
if 'bmstu_loaded' not in st.session_state:
    st.session_state.bmstu_loaded = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'current_rank' not in st.session_state:
    st.session_state.current_rank = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

# Загрузка CSV файла
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV файл с данными вузов", 
    type=['csv'],
    help="Файл должен содержать все 42 признака в отдельных колонках",
    key="file_uploader"  # Добавляем ключ для отслеживания изменений
)

# Обработка загруженного CSV файла
if uploaded_file is not None and not st.session_state.file_processed:
    result = process_csv_file(uploaded_file)
    if result:
        csv_data, full_df = result
        st.session_state.csv_data = csv_data
        st.session_state.file_processed = True
        st.sidebar.success("✅ Данные из CSV готовы к использованию")
        
        # Показываем превью данных
        if st.sidebar.checkbox("Показать превью данных"):
            st.sidebar.write("**Первые 5 записей:**")
            st.sidebar.dataframe(full_df.head())

# Кнопка для использования данных из CSV
if st.session_state.csv_data and st.sidebar.button("📊 Использовать данные из CSV"):
    st.session_state.use_csv = True
    st.session_state.file_processed = False  # Сбрасываем для возможности загрузки нового файла
    st.rerun()

# Функция для получения значений по умолчанию с учетом CSV данных
def get_form_default(feat):
    if st.session_state.use_csv and feat in st.session_state.csv_data:
        return st.session_state.csv_data[feat]
    
    # Значения по умолчанию для разных типов признаков
    if "egescore" in feat:
        return 60.0
    elif "olympiad" in feat:
        return 10
    elif feat == "competition":
        return 5.0
    elif "share" in feat or "percent" in feat:
        return 10.0
    elif "aspirants" in feat:
        return 2.0
    elif feat == "foreign_professors":
        return 2
    elif feat == "niokr_total":
        return 50000.0
    elif feat == "niokr_per_npr":
        return 200.0
    elif "publications" in feat:
        return 100
    elif "citations" in feat:
        return 500
    elif "income" in feat or "salary" in feat:
        return 100000.0
    elif feat == "journals_published":
        return 2
    elif feat == "grants_per_100_npr":
        return 5.0
    elif feat == "npr_per_100_students":
        return 8.0
    elif feat == "lib_books_per_student":
        return 100
    elif feat == "area_per_student":
        return 15.0
    elif feat == "pc_per_student":
        return 0.5
    else:
        return 10.0

# Отображаем информацию о загруженных данных перед формой
if st.session_state.use_csv and st.session_state.csv_data:
    st.info("📊 Используются данные из загруженного CSV файла")

# Форма ввода данных
with st.form("input_form"):
    st.write("Введите данные по вузу:")
    input_data = {}
    
    # Группировка признаков для лучшего UX
    st.subheader("📊 Академические показатели")
    academic_features = [
        'egescore_avg', 'egescore_contract', 'egescore_min', 
        'olympiad_winners', 'olympiad_other', 'competition'
    ]
    for feat in academic_features:
        if feat in feature_order:
            default_val = get_form_default(feat)
            
            if "egescore" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 120.0, float(default_val), step=0.1, 
                                            key=f"slider_{feat}",  # Уникальный ключ для каждого элемента
                                            help="Максимум 120 для учета олимпиадников с 100+ баллами")
            elif "olympiad" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=f"num_{feat}",
                                                help="До 5000 человек для крупных вузов")
            elif feat == "competition":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                            key=f"slider_competition",
                                            help="Конкурс может достигать 100 человек на место в престижных вузах")

    st.subheader("🎯 Целевой прием и магистратура")
    target_features = [
        'target_admission_share', 'target_contract_in_tech',
        'magistracy_share', 'aspirantura_share', 'external_masters', 
        'external_grad_share', 'aspirants_per_100_students'
    ]
    for feat in target_features:
        if feat in feature_order:
            default_val = get_form_default(feat)
            
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_{feat}",
                                            help="Может превышать 100% для специализированных программ")
            elif feat == "aspirants_per_100_students":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                                key=f"num_aspirants",
                                                help="До 100 аспирантов на 100 студентов для исследовательских вузов")
            elif feat == "external_masters":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_external_masters",
                                            help="Может превышать 100% для программ переподготовки")
            elif feat == "target_contract_in_tech":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_target_contract",
                                            help="Может превышать 100% для технических специализаций")

    # Продолжить для остальных групп признаков аналогичным образом...
    # [Остальной код групп признаков остается таким же, но с использованием get_form_default(feat)]

    # Кнопка отправки формы
    submitted = st.form_submit_button("🔢 Предсказать место")

# Обработка предсказания (остальной код остается без изменений)
if submitted and predictor is not None:
    # Заполняем недостающие признаки значениями по умолчанию
    current_data = {}
    for feat in feature_order:
        if feat in input_data:
            current_data[feat] = input_data[feat]
        else:
            if "share" in feat or "percent" in feat:
                current_data[feat] = 10.0
            else:
                current_data[feat] = 100.0
    
    input_data = current_data
    
    # Проверяем, что все признаки присутствуют
    missing_features = set(feature_order) - set(input_data.keys())
    if missing_features:
        st.error(f"❌ Не заполнены следующие признаки: {missing_features}")
        st.info("Пожалуйста, заполните все поля формы")
        st.stop()

    st.session_state.input_data = input_data
    st.session_state.submitted = True
    st.session_state.use_csv = False  # Сбрасываем флаг использования CSV после отправки
    
    user_df = pd.DataFrame([input_data])
    user_df = user_df[feature_order]
    
    with st.spinner("Вычисляем рейтинг..."):
        try:
            rank = predictor.predict_rank(user_df)
            st.session_state.current_rank = rank
            st.success(f"🏆 Предсказанное место: **{rank:.1f}**")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

# [Остальной код рекомендаций и сайдбара остается без изменений]

# Информация о модели в сайдбаре
with st.sidebar:
    st.header("📁 Формат CSV файла")
    st.write("""
    CSV файл должен содержать колонки со следующими названиями:
    - Все 42 признака из списка
    - Данные в числовом формате
    - Первая строка - заголовки
    - Кодировка: UTF-8 или Windows-1251
    """)
    
    # Кнопка для скачивания шаблона CSV
    @st.cache_data
    def create_template_csv():
        template_df = pd.DataFrame(columns=feature_order)
        template_df.loc[0] = [0] * len(feature_order)
        return template_df.to_csv(index=False, encoding='utf-8')
    
    template_csv = create_template_csv()
    st.download_button(
        label="📥 Скачать шаблон CSV",
        data=template_csv,
        file_name="raex_template.csv",
        mime="text/csv",
        help="Скачайте шаблон для заполнения данными"
    )
    
    # Кнопка для сброса формы
    if st.button("🔄 Сбросить форму"):
        for key in list(st.session_state.keys()):
            if key not in ['_rerun', '_pages']:
                del st.session_state[key]
        st.rerun()
    
    # Показать все необходимые признаки
    if st.checkbox("Показать все необходимые признаки"):
        st.write("**Всего признаков:**", len(feature_order))
        for i, feat in enumerate(feature_order, 1):
            st.write(f"{i}. {russian_name(feat)} ({feat})")