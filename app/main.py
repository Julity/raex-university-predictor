# app/main.py
import streamlit as st
import pandas as pd
import sys
import os
import io

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

# Данные для ДГТУ и ДонНТУ
DGSU_DATA = {
    # Академические показатели
    'egescore_avg': 64.13, 'egescore_min': 45.26, 'egescore_contract': 64.13,
    'olympiad_winners': 0, 'olympiad_other': 1, 'competition': 3.0,
    # Целевой прием и магистратура
    'target_admission_share': 1.44, 'target_contract_in_tech': 1.99,
    'magistracy_share': 13.32, 'aspirantura_share': 2.65,
    'external_masters': 19.62, 'external_grad_share': 52.66,
    'aspirants_per_100_students': 2.65,
    # Международная деятельность
    'foreign_students_share': 8.53, 'foreign_non_cis': 6.34, 'foreign_cis': 2.19,
    'foreign_graduated': 11.19, 'mobility_outbound': 0.21,
    'foreign_staff_share': 0.11, 'foreign_professors': 4,
    # Научная деятельность
    'niokr_total': 636449.5, 'niokr_share_total': 7.53, 'niokr_own_share': 97.25,
    'niokr_per_npr': 361.38, 'scopus_publications': 0, 'risc_publications': 122.42,
    'risc_citations': 346.76, 'foreign_niokr_income': 0, 'journals_published': 10,
    'grants_per_100_npr': 1.53,
    # Финансовые показатели
    'foreign_edu_income': 155646.5, 'total_income_per_student': 401.42,
    'self_income_per_npr': 1195.27, 'self_income_share': 25.56,
    'ppc_salary_index': 208.17, 'avg_salary_grads': 82740,
    # Инфраструктура и кадры
    'npr_with_degree_percent': 65.66, 'npr_per_100_students': 3.81,
    'young_npr_share': 12.5, 'lib_books_per_student': 70.44,
    'area_per_student': 8.46, 'pc_per_student': 0.18
}

DONNTU_DATA = {
    # Академические показатели
    'egescore_avg': 79.10, 'egescore_contract': 70.74, 'egescore_min': 69.00,
    'olympiad_winners': 0, 'olympiad_other': 2, 'competition': 5.0,
    # Целевой прием и магистратура
    'target_admission_share': 0.00, 'target_contract_in_tech': 0.00,
    'magistracy_share': 21.72, 'aspirantura_share': 2.76,
    'external_masters': 7.35, 'external_grad_share': 91.03,
    'aspirants_per_100_students': 4.21,
    # Международная деятельность
    'foreign_students_share': 0.06, 'foreign_non_cis': 0.00, 'foreign_cis': 0.06,
    'foreign_graduated': 0.26, 'mobility_outbound': 0.00,
    'foreign_staff_share': 0.00, 'foreign_professors': 0,
    # Научная деятельность
    'niokr_total': 56943.10, 'niokr_share_total': 3.27, 'niokr_own_share': 0.00,
    'niokr_per_npr': 134.02, 'scopus_publications': 150, 'risc_publications': 25.42,
    'risc_citations': 890.09, 'foreign_niokr_income': 0.00, 'journals_published': 5,
    'grants_per_100_npr': 0.00,
    # Финансовые показатели
    'foreign_edu_income': 0.00, 'total_income_per_student': 494.30,
    'self_income_per_npr': 185.09, 'self_income_share': 4.54,
    'ppc_salary_index': 0.00, 'avg_salary_grads': 75000,
    # Инфраструктура и кадры
    'npr_with_degree_percent': 60.54, 'npr_per_100_students': 5.28,
    'young_npr_share': 6.94, 'lib_books_per_student': 346.45,
    'area_per_student': 33.71, 'pc_per_student': 0.83
}

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
        
        # Выбираем первую строку (первый вуз) для предсказания
        sample_data = df.iloc[0][feature_order].to_dict()
        
        st.success(f"✅ Файл успешно загружен! Записей: {len(df)}")
        st.info(f"📝 Используются данные первого вуза из файла")
        
        return sample_data, df
    
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
        return None

# Инициализация session_state
# Инициализация session_state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = {}
if 'use_csv' not in st.session_state:
    st.session_state.use_csv = False
if 'university_loaded' not in st.session_state:
    st.session_state.university_loaded = None
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0  # Ключ для принудительного обновления формы


# Обработка кнопок университетов
col1, col2 = st.columns(2)
with col1:
    if st.button("🏛️ Заполнить данные ДГТУ", type="primary", use_container_width=True, key="btn_dgsu"):
        st.session_state.csv_data = DGSU_DATA
        st.session_state.use_csv = True
        st.session_state.university_loaded = "ДГТУ"
        st.session_state.form_key += 1  # Изменяем ключ формы
        st.success("✅ Данные ДГТУ загружены!")

with col2:
    if st.button("🎓 Заполнить данные ДонНТУ", type="secondary", use_container_width=True, key="btn_donntu"):
        st.session_state.csv_data = DONNTU_DATA
        st.session_state.use_csv = True
        st.session_state.university_loaded = "ДонНТУ"
        st.session_state.form_key += 1  # Изменяем ключ формы
        st.success("✅ Данные ДонНТУ загружены!")

st.markdown("---")

# Загрузка CSV файла
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV файл с данными вузов", 
    type=['csv'],
    help="Файл должен содержать все 42 признака в отдельных колонках"
)

# Обработка загруженного CSV файла
if uploaded_file is not None:
    result = process_csv_file(uploaded_file)
    if result:
        csv_data, full_df = result
        st.session_state.csv_data = csv_data
        st.session_state.csv_loaded = True
        st.session_state.university_loaded = "из CSV файла"
        st.sidebar.success("✅ Данные из CSV готовы к использованию")
        
        # Показываем превью данных
        if st.sidebar.checkbox("Показать превью данных"):
            st.sidebar.write("**Первые 5 записей:**")
            st.sidebar.dataframe(full_df.head())

# Кнопка для использования данных из CSV
if st.sidebar.button("📊 Использовать данные из CSV", type="primary"):
    if st.session_state.csv_loaded:
        st.session_state.use_csv = True
        st.session_state.force_rerun = True
        st.rerun()
    else:
        st.sidebar.warning("❌ Сначала загрузите CSV файл")

# Функция для получения значения по умолчанию с учетом CSV данных
def get_default_value(feat, csv_defaults, use_csv_data):
    if use_csv_data and feat in csv_defaults:
        return csv_defaults[feat]
    
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

# Сбрасываем флаг после rerun
if st.session_state.get('force_rerun', False):
    st.session_state.force_rerun = False

form_key = f"input_form_{st.session_state.form_key}"

# Форма ввода данных
with st.form(form_key):
    st.write("Введите данные по вузу:")
    input_data = {}
    
    # Если есть данные из CSV, используем их как значения по умолчанию
    use_csv_data = st.session_state.get("use_csv", False)
    csv_defaults = st.session_state.get("csv_data", {})
    
    # Отображаем информацию о загруженных данных
    if use_csv_data and csv_defaults:
        university_name = st.session_state.get("university_loaded", "загруженного CSV файла")
        st.info(f"📊 Используются данные {university_name}")
    
    # Группировка признаков для лучшего UX
    st.subheader("📊 Академические показатели")
    academic_features = [
        'egescore_avg', 'egescore_contract', 'egescore_min', 
        'olympiad_winners', 'olympiad_other', 'competition'
    ]
    for feat in academic_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"  # Уникальный ключ для каждого виджета
            if "egescore" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 120.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Максимум 120 для учета олимпиадников с 100+ баллами")
            elif "olympiad" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=widget_key,
                                                help="До 5000 человек для крупных вузов")
            elif feat == "competition":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Конкурс может достигать 100 человек на место в престижных вузах")

    st.subheader("🎯 Целевой прием и магистратура")
    target_features = [
        'target_admission_share', 'target_contract_in_tech',
        'magistracy_share', 'aspirantura_share', 'external_masters', 
        'external_grad_share', 'aspirants_per_100_students'
    ]
    for feat in target_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для специализированных программ")
            elif feat == "aspirants_per_100_students":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                                key=widget_key,
                                                help="До 100 аспирантов на 100 студентов для исследовательских вузов")
            elif feat == "external_masters":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для программ переподготовки")
            elif feat == "target_contract_in_tech":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для технических специализаций")

    st.subheader("🌍 Международная деятельность")
    international_features = [
        'foreign_students_share', 'foreign_non_cis', 'foreign_cis', 
        'foreign_graduated', 'mobility_outbound', 'foreign_staff_share', 
        'foreign_professors'
    ]
    for feat in international_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для международных программ")
            elif feat == "foreign_professors":
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=widget_key,
                                                help="До 5000 иностранных преподавателей для крупных международных вузов")
            elif feat == "foreign_non_cis":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Доля иностранцев вне СНГ может быть высокой в международных вузах")
            elif feat == "foreign_cis":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Доля иностранцев из СНГ")
            elif feat == "foreign_graduated":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Доля окончивших иностранцев")
            elif feat == "mobility_outbound":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% при множественных стажировках")

    st.subheader("🔬 Научная деятельность")
    research_features = [
        'niokr_total', 'niokr_share_total', 'niokr_own_share', 'niokr_per_npr',
        'scopus_publications', 'risc_publications', 'risc_citations',
        'foreign_niokr_income', 'journals_published', 'grants_per_100_npr'
    ]
    for feat in research_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для исследовательских центров")
            elif feat == "niokr_total":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 50000000.0, float(default_val), step=100000.0, 
                                                key=widget_key,
                                                help="До 50 млн руб. для крупных исследовательских проектов")
            elif feat == "niokr_per_npr":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500000.0, float(default_val), step=1000.0, 
                                                key=widget_key,
                                                help="До 500 тыс. руб. на преподавателя в ведущих научных центрах")
            elif "publications" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0, 100000, int(default_val), 
                                                key=widget_key,
                                                help="До 100000 публикаций для крупных исследовательских университетов")
            elif "citations" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0, 1000000, int(default_val), 
                                                key=widget_key,
                                                help="До 1 млн цитирований для ведущих научных школ")
            elif "income" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100000000.0, float(default_val), step=100000.0, 
                                                key=widget_key,
                                                help="До 100 млн руб. доходов от международных исследований")
            elif feat == "journals_published":
                input_data[feat] = st.number_input(russian_name(feat), 0, 500, int(default_val), 
                                                key=widget_key,
                                                help="До 500 журналов для крупных издательских центров")
            elif feat == "grants_per_100_npr":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                                key=widget_key,
                                                help="До 500 грантов на 100 преподавателей в исследовательских вузах")

    st.subheader("💰 Финансовые показатели")
    financial_features = [
        'foreign_edu_income', 'total_income_per_student', 'self_income_per_npr',
        'self_income_share', 'ppc_salary_index', 'avg_salary_grads'
    ]
    for feat in financial_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"
            if "share" in feat or "percent" in feat or "index" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                            key=widget_key,
                                            help="Может достигать 500% для высокооплачиваемых специальностей")
            elif "income" in feat or "salary" in feat:
                max_val = 10000000.0 if feat == "total_income_per_student" else 5000000.0
                step_val = 10000.0 if feat == "total_income_per_student" else 5000.0
                input_data[feat] = st.number_input(russian_name(feat), 0.0, max_val, float(default_val), step=step_val, 
                                                key=widget_key,
                                                help=f"До {max_val:,.0f} руб. для ведущих вузов с высокими доходами")

    st.subheader("🏫 Инфраструктура и кадры")
    infrastructure_features = [
        'npr_with_degree_percent', 'npr_per_100_students', 'young_npr_share',
        'lib_books_per_student', 'area_per_student', 'pc_per_student'
    ]
    for feat in infrastructure_features:
        if feat in feature_order:
            default_val = get_default_value(feat, csv_defaults, use_csv_data)
            widget_key = f"{feat}_{st.session_state.form_key}"
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=widget_key,
                                            help="Может превышать 100% для специализированных кафедр")
            elif feat == "npr_per_100_students":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                                key=widget_key,
                                                help="До 100 преподавателей на 100 студентов в магистратуре/аспирантуре")
            elif feat == "lib_books_per_student":
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=widget_key,
                                                help="До 5000 книг на студента в вузах с богатыми библиотеками")
            elif feat == "area_per_student":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                                key=widget_key,
                                                help="До 500 м² на студента в кампусных университетах")
            elif feat == "pc_per_student":
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 10.0, float(default_val), step=0.1, 
                                                key=widget_key,
                                                help="До 10 компьютеров на студента в IT-вузах")
    
    # Кнопка отправки формы
    submitted = st.form_submit_button("🔢 Прогноз")

# Проверка заполнения всех признаков
if submitted:
    # Создаем DataFrame с текущими данными
    current_data = {}
    for feat in feature_order:
        if feat in input_data:
            current_data[feat] = input_data[feat]
        else:
            # Если признак отсутствует, устанавливаем значение по умолчанию
            if "share" in feat or "percent" in feat:
                current_data[feat] = 10.0
            else:
                current_data[feat] = 100.0
    
    # Обновляем input_data всеми признаками
    input_data = current_data
    
    # Проверяем, что все признаки присутствуют
    missing_features = set(feature_order) - set(input_data.keys())
    if missing_features:
        st.error(f"❌ Не заполнены следующие признаки: {missing_features}")
        st.info("Пожалуйста, заполните все поля формы")
        st.stop()
if st.sidebar.button("🔄 Сбросить форму", key="btn_reset"):
    st.session_state.use_csv = False
    st.session_state.csv_data = {}
    st.session_state.university_loaded = None
    st.session_state.form_key += 1
    st.session_state.submitted = False
    st.sidebar.success("✅ Форма сброшена!")
# Обработка предсказания
if submitted and predictor is not None:
    st.session_state["input_data"] = input_data
    st.session_state["submitted"] = True
    st.session_state["use_csv"] = False  # Сбрасываем флаг использования CSV
    st.session_state["bmstu_loaded"] = False  # Сбрасываем флаг Бауманки
    st.session_state["university_loaded"] = None
    
    user_df = pd.DataFrame([input_data])
    
    # Убедимся, что все признаки присутствуют и в правильном порядке
    for feat in feature_order:
        if feat not in user_df.columns:
            st.error(f"Отсутствует признак: {feat}")
            st.stop()
    
    # Переупорядочиваем колонки согласно feature_order
    user_df = user_df[feature_order]
    
    with st.spinner("Вычисляем рейтинг..."):
        try:
            rank = predictor.predict_rank(user_df)
            st.session_state["current_rank"] = rank
            st.success(f"🏆 Предсказанное место: **{rank:.1f}**")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.write("Проверьте, что все поля заполнены корректно")

# Рекомендации по улучшению
if st.session_state.get("submitted", False) and predictor is not None and "current_rank" in st.session_state:
    st.markdown("---")
    st.subheader("🎯 Улучшение позиции")
    
    current_rank = st.session_state["current_rank"]
    st.write(f"Текущий ранг: **{current_rank:.1f}**")
    
    desired_top = st.slider("В какой топ вы хотите попасть?", 1, 1000, min(20, int(current_rank)), key="desired_top")
    
    # Разрешённые для рекомендаций признаки
    improvement_options = {
        "Академические показатели": [
            'egescore_avg', 'olympiad_winners', 'competition',
            'target_admission_share', 'magistracy_share', 'external_masters'
        ],
        "Международная деятельность": [
            'foreign_students_share', 'foreign_professors', 'mobility_outbound',
            'foreign_edu_income', 'foreign_niokr_income'
        ],
        "Научно-исследовательская деятельность": [
            'scopus_publications', 'niokr_total', 'grants_per_100_npr',
            'journals_published', 'risc_publications'
        ],
        "Финансовые показатели": [
            'total_income_per_student', 'self_income_per_npr', 'self_income_share'
        ],
        "Инфраструктура и кадры": [
            'npr_with_degree_percent', 'young_npr_share', 'area_per_student',
            'pc_per_student', 'lib_books_per_student'
        ]
    }
    
    st.markdown("Выберите группы признаков для улучшения:")
    selected_groups = st.multiselect(
        "Группы показателей",
        options=list(improvement_options.keys()),
        default=list(improvement_options.keys())
    )
    
    # Собираем все разрешённые признаки из выбранных групп
    allowed_features = []
    for group in selected_groups:
        allowed_features.extend(improvement_options[group])
    
    if st.button("🔄 Найти рекомендации по улучшению", key="improve_btn"):
        user_df = pd.DataFrame([st.session_state["input_data"]])
        user_df = user_df[feature_order]
        
        with st.spinner("Анализируем возможные улучшения..."):
            try:
                result = predictor.suggest_improvement(
                    user_df,
                    desired_top,
                    current_rank=current_rank,
                    allowed_features=allowed_features
                )
                
                if len(result) == 2:
                    recommendations, improved_rank = result
                    percent_changes = []
                else:
                    recommendations, improved_rank, percent_changes = result
                
                st.markdown("### Рекомендации по улучшению:")
                
                if improved_rank <= desired_top:
                    st.success(f"🎉 Можно достичь топа-{desired_top}! Предсказанный ранг после улучшений: {improved_rank:.1f}")
                else:
                    st.warning(f"⚠️ Полное достижение топа-{desired_top} может быть сложным. Предсказанный ранг после улучшений: {improved_rank:.1f}")
                
                if recommendations:
                    st.markdown("📈 Рекомендации:")
                    meaningful_count = 0
                    
                    for i, recommendation in enumerate(recommendations, 1):
                        if len(recommendation) == 3:
                            feat, old, new = recommendation
                            if old > 0:
                                percent_change = ((new - old) / old * 100)
                            else:
                                percent_change = 100 if new > 0 else 0
                        elif len(recommendation) == 4:
                            feat, old, new, percent_change = recommendation
                        else:
                            continue
                        
                        if abs(percent_change) < 0.01 or abs(new - old) < 0.1:
                            continue
                        
                        meaningful_count += 1
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(f"**{meaningful_count}. {russian_name(feat)}**")
                        with col2:
                            st.write(f"`{old:.2f} → {new:.2f}`")
                        with col3:
                            st.write(f"`({percent_change:+.1f}%)`")
                        
                        progress_value = min(100, max(0, percent_change / 2 + 50))
                        st.progress(progress_value / 100)
                    
                    if meaningful_count == 0:
                        st.info("ℹ️ Не найдено значимых рекомендаций для улучшения. Попробуйте выбрать другие группы показателей.")
                else:
                    st.info("ℹ️ Не найдено конкретных рекомендаций для улучшения.")
                    
            except Exception as e:
                st.error(f"Ошибка при генерации рекомендаций: {e}")

elif predictor is None:
    st.error("❌ Модель не загружена. Проверьте наличие обученных моделей в папке 'models/'")

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
        st.session_state.use_csv = False
        st.session_state.csv_data = {}
        st.session_state.bmstu_loaded = False
        st.session_state.submitted = False
        st.session_state.csv_loaded = False
        st.session_state.university_loaded = None
        st.rerun()
    
    # Показать все необходимые признаки
    if st.checkbox("Показать все необходимые признаки"):
        st.write("Всего признаков:", len(feature_order))
        for i, feat in enumerate(feature_order, 1):
            st.write(f"{i}. {russian_name(feat)} ({feat})")