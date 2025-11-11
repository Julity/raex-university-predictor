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

# Загрузка CSV файла
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV файл с данными вузов", 
    type=['csv'],
    help="Файл должен содержать все 42 признака в отдельных колонках"
)

# Инициализация session_state для данных CSV
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = {}
if 'use_csv' not in st.session_state:
    st.session_state.use_csv = False
if 'bmstu_loaded' not in st.session_state:
    st.session_state.bmstu_loaded = False
# ДОБАВЬТЕ в начало инициализации session_state:
if 'form_initialized' not in st.session_state:
    st.session_state.form_initialized = False
if 'csv_applied' not in st.session_state:
    st.session_state.csv_applied = False

# ЗАМЕНИТЕ весь блок CSV обработки на:
# Обработка загруженного CSV файла
if uploaded_file is not None:
    # Сбрасываем флаги при новой загрузке файла
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.use_csv = False
        st.session_state.bmstu_loaded = False
        st.session_state.csv_applied = False
        st.session_state.form_initialized = False
        st.session_state.current_file = uploaded_file.name
    
    result = process_csv_file(uploaded_file)
    if result:
        csv_data, full_df = result
        st.session_state.csv_data = csv_data
        st.sidebar.success("✅ Файл успешно загружен!")
        
        # Показываем превью
        if st.sidebar.checkbox("Показать превью данных"):
            st.sidebar.write("**Первые 5 записей:**")
            st.sidebar.dataframe(full_df.head())
        
        # Автоматически применяем данные если они еще не применены
        if not st.session_state.csv_applied:
            st.session_state.use_csv = True
            st.session_state.csv_applied = True
            st.session_state.form_initialized = False
            st.rerun()
# # Кнопка для заполнения данными Бауманки в сайдбаре
# if st.sidebar.button("🎯 Заполнить данные МГТУ им. Баумана (2023)"):
#     # Данные для Бауманки
#     bmstu_data = {
#         # Академические показатели
#         'egescore_avg': 80.83, 'egescore_contract': 71.98, 'egescore_min': 54.55,
#         'olympiad_winners': 8, 'olympiad_other': 236, 'competition': 5.0,
#         # Целевой прием и магистратура
#         'target_admission_share': 13.59, 'target_contract_in_tech': 20.37,
#         'magistracy_share': 10.30, 'aspirantura_share': 2.70,
#         'external_masters': 98.72, 'external_grad_share': 47.70,
#         'aspirants_per_100_students': 3.70,
#         # Международная деятельность
#         'foreign_students_share': 5.71, 'foreign_non_cis': 3.70, 'foreign_cis': 2.01,
#         'foreign_graduated': 7.66, 'mobility_outbound': 0.07,
#         'foreign_staff_share': 0.22, 'foreign_professors': 0,
#         # Научная деятельность
#         'niokr_total': 3982904.40, 'niokr_share_total': 22.40, 'niokr_own_share': 84.29,
#         'niokr_per_npr': 1919.01, 'scopus_publications': 160.44, 'risc_publications': 160.44,
#         'risc_citations': 409.68, 'foreign_niokr_income': 0.00, 'journals_published': 13,
#         'grants_per_100_npr': 2.84,
#         # Финансовые показатели
#         'foreign_edu_income': 31664.10, 'total_income_per_student': 827.28,
#         'self_income_per_npr': 1939.98, 'self_income_share': 22.59,
#         'ppc_salary_index': 200.57, 'avg_salary_grads': 100.0,
#         # Инфраструктура и кадры
#         'npr_with_degree_percent': 62.89, 'npr_per_100_students': 5.77,
#         'young_npr_share': 13.63, 'lib_books_per_student': 106.41,
#         'area_per_student': 10.36, 'pc_per_student': 0.36
#     }
    
#     st.session_state.csv_data = bmstu_data
#     st.session_state.use_csv = True
#     st.session_state.bmstu_loaded = True
#     st.rerun()
# После превью данных ДОБАВЬТЕ:

# Форма ввода данных
# Принудительное обновление формы при применении CSV данных
if st.session_state.get('use_csv', False) and st.session_state.get('csv_data'):
    if not st.session_state.get('form_initialized', False):
        st.session_state.form_initialized = True
        st.rerun()

# Форма ввода данных
with st.form("input_form"):
    # Показываем статус применения данных
    if st.session_state.get("use_csv", False) and st.session_state.get("csv_data", {}):
        st.info("📊 Используются данные из загруженного CSV файла")
    
    st.write("Введите данные по вузу:")
    input_data = {}
    
    # Если есть данные из CSV, используем их как значения по умолчанию
    use_csv_data = st.session_state.get("use_csv", False)
    csv_defaults = st.session_state.get("csv_data", {})
    
 
    # Отображаем информацию о загруженных данных
    if use_csv_data and csv_defaults:
        if st.session_state.get("bmstu_loaded", False):
            st.info("🎯 Используются данные МГТУ им. Баумана за 2023 год")
        else:
            st.info("📊 Используются данные из загруженного CSV файла")
    
        # Группировка признаков для лучшего UX
    st.subheader("📊 Академические показатели")
    academic_features = [
        'egescore_avg', 'egescore_contract', 'egescore_min', 
        'olympiad_winners', 'olympiad_other', 'competition'
    ]
    for feat in academic_features:
        if feat in feature_order:
            default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!            if "egescore" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 120.0, float(default_val), step=0.1, key=f"slider_academic_{feat}", help="Максимум 120 для учета олимпиадников с 100+ баллами")
            elif "olympiad" in feat:
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=f"num_academic_{feat}",
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
            default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_target_{feat}",
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

    st.subheader("🌍 Международная деятельность")
    international_features = [
        'foreign_students_share', 'foreign_non_cis', 'foreign_cis', 
        'foreign_graduated', 'mobility_outbound', 'foreign_staff_share', 
        'foreign_professors'
    ]
    for feat in international_features:
        if feat in feature_order:
            default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!
            if "share" in feat or "percent" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=f"slider_int_{feat}",
                                            help="Может превышать 100% для международных программ")
            elif feat == "foreign_professors":
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=f"num_foreign_professors",
                                                help="До 5000 иностранных преподавателей для крупных международных вузов")
            elif feat == "foreign_non_cis":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=f"slider_foreign_non_cis",
                                            help="Доля иностранцев вне СНГ может быть высокой в международных вузах")
            elif feat == "foreign_cis":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=f"slider_foreign_cis",
                                            help="Доля иностранцев из СНГ")
            elif feat == "foreign_graduated":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=f"slider_foreign_graduated",
                                            help="Доля окончивших иностранцев")
            elif feat == "mobility_outbound":
                input_data[feat] = st.slider(russian_name(feat), 0.0, 150.0, float(default_val), step=0.1, 
                                            key=f"slider_mobility_outbound",
                                            help="Может превышать 100% при множественных стажировках")

    st.subheader("🔬 Научная деятельность")
    research_features = [
        'niokr_total', 'niokr_share_total', 'niokr_own_share', 'niokr_per_npr',
        'scopus_publications', 'risc_publications', 'risc_citations',
        'foreign_niokr_income', 'journals_published', 'grants_per_100_npr'
    ]
    for feat in research_features:
        if feat in feature_order:
            if "share" in feat or "percent" in feat:
                default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_research_{feat}",
                                            help="Может превышать 100% для исследовательских центров")
            elif feat == "niokr_total":
                default_val = csv_defaults.get(feat, 50000.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 50000000.0, float(default_val), step=100000.0, 
                                                key=f"num_niokr_total",
                                                help="До 50 млн руб. для крупных исследовательских проектов")
            elif feat == "niokr_per_npr":
                default_val = csv_defaults.get(feat, 200.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500000.0, float(default_val), step=1000.0, 
                                                key=f"num_niokr_per_npr",
                                                help="До 500 тыс. руб. на преподавателя в ведущих научных центрах")
            elif "publications" in feat:
                default_val = csv_defaults.get(feat, 100)
                max_val = 100000 if feat == "scopus_publications" else 500000
                input_data[feat] = st.number_input(russian_name(feat), 0, max_val, int(default_val), 
                                                key=f"num_{feat}",
                                                help=f"До {max_val} публикаций для крупных исследовательских университетов")
            elif "citations" in feat:
                default_val = csv_defaults.get(feat, 500)
                input_data[feat] = st.number_input(russian_name(feat), 0, 1000000, int(default_val), 
                                                key=f"num_{feat}",
                                                help="До 1 млн цитирований для ведущих научных школ")
            elif "income" in feat:
                default_val = csv_defaults.get(feat, 10000.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100000000.0, float(default_val), step=100000.0, 
                                                key=f"num_{feat}",
                                                help="До 100 млн руб. доходов от международных исследований")
            elif feat == "journals_published":
                default_val = csv_defaults.get(feat, 2)
                input_data[feat] = st.number_input(russian_name(feat), 0, 500, int(default_val), 
                                                key=f"num_journals",
                                                help="До 500 журналов для крупных издательских центров")
            elif feat == "grants_per_100_npr":
                default_val = csv_defaults.get(feat, 5.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                                key=f"num_grants",
                                                help="До 500 грантов на 100 преподавателей в исследовательских вузах")

    st.subheader("💰 Финансовые показатели")
    financial_features = [
        'foreign_edu_income', 'total_income_per_student', 'self_income_per_npr',
        'self_income_share', 'ppc_salary_index', 'avg_salary_grads'
    ]
    for feat in financial_features:
        if feat in feature_order:
            default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!
            if "share" in feat or "percent" in feat or "index" in feat:
                input_data[feat] = st.slider(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                            key=f"slider_finance_{feat}",
                                            help="Может достигать 500% для высокооплачиваемых специальностей")
            elif "income" in feat or "salary" in feat:
                max_val = 10000000.0 if feat == "total_income_per_student" else 5000000.0
                step_val = 10000.0 if feat == "total_income_per_student" else 5000.0
                input_data[feat] = st.number_input(russian_name(feat), 0.0, max_val, float(default_val), step=step_val, 
                                                key=f"num_finance_{feat}",
                                                help=f"До {max_val:,.0f} руб. для ведущих вузов с высокими доходами")

    st.subheader("🏫 Инфраструктура и кадры")
    infrastructure_features = [
        'npr_with_degree_percent', 'npr_per_100_students', 'young_npr_share',
        'lib_books_per_student', 'area_per_student', 'pc_per_student'
    ]
    for feat in infrastructure_features:
        if feat in feature_order:
            if "share" in feat or "percent" in feat:
                default_val = get_default(feat, csv_defaults, use_csv_data)  # ВСЕГО ОДНА СТРОКА!
                input_data[feat] = st.slider(russian_name(feat), 0.0, 200.0, float(default_val), step=0.1, 
                                            key=f"slider_infra_{feat}",
                                            help="Может превышать 100% для специализированных кафедр")
            elif feat == "npr_per_100_students":
                default_val = csv_defaults.get(feat, 8.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 100.0, float(default_val), step=0.1, 
                                                key=f"num_npr_per_100",
                                                help="До 100 преподавателей на 100 студентов в магистратуре/аспирантуре")
            elif feat == "lib_books_per_student":
                default_val = csv_defaults.get(feat, 100)
                input_data[feat] = st.number_input(russian_name(feat), 0, 5000, int(default_val), 
                                                key=f"num_lib_books",
                                                help="До 5000 книг на студента в вузах с богатыми библиотеками")
            elif feat == "area_per_student":
                default_val = csv_defaults.get(feat, 15.0)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 500.0, float(default_val), step=1.0, 
                                                key=f"num_area",
                                                help="До 500 м² на студента в кампусных университетах")
            elif feat == "pc_per_student":
                default_val = csv_defaults.get(feat, 0.5)
                input_data[feat] = st.number_input(russian_name(feat), 0.0, 10.0, float(default_val), step=0.1, 
                                                key=f"num_pc",
                                                help="До 10 компьютеров на студента в IT-вузах")
    # Кнопка отправки формы
    submitted = st.form_submit_button("🔢 Предсказать место")

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

# Обработка предсказания
if submitted and predictor is not None:
    st.session_state["input_data"] = input_data
    st.session_state["submitted"] = True
    st.session_state["use_csv"] = False  # Сбрасываем флаг использования CSV
    st.session_state["bmstu_loaded"] = False  # Сбрасываем флаг Бауманки
    
    

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
            'egescore_avg',           #  СРЕДНИЙ БАЛЛ ЕГЭ - можно повысить через профориентацию, подготовительные курсы
            'olympiad_winners',       #  ПОБЕДИТЕЛИ ОЛИМПИАД - активная работа со школьными олимпиадами, создание своих олимпиад
            'competition',            #  КОНКУРС НА МЕСТО - маркетинг, привлекательные программы, увеличение плана приема
            'target_admission_share', #  ЦЕЛЕВОЙ ПРИЕМ - сотрудничество с предприятиями, госзаказ
            'magistracy_share',       #  ДОЛЯ МАГИСТРАТУРЫ - развитие магистерских программ
            'external_masters'        #  МАГИСТРАНТЫ ИЗ ДРУГИХ ВУЗОВ - программы переподготовки, сетевые программы
        ],
        "Международная деятельность": [
            'foreign_students_share',    #  ИНОСТРАННЫЕ СТУДЕНТЫ - рекрутинг, англоязычные программы, визовая поддержка
            'foreign_professors',        #  ИНОСТРАННЫЕ ПРОФЕССОРА - программы приглашения, международные гранты
            'mobility_outbound',         #  СТАЖИРОВКИ ЗА РУБЕЖОМ - партнерства с зарубежными вузами, программы обмена
            'foreign_edu_income',        #  ДОХОДЫ ОТ ИНОСТРАНЦЕВ - платное образование для иностранцев
            'foreign_niokr_income'       #  МЕЖДУНАРОДНЫЕ НИОКР - участие в международных исследовательских проектах
        ],
        "Научно-исследовательская деятельность": [
            'scopus_publications',      # ПУБЛИКАЦИИ SCOPUS - гранты на публикации, мотивация преподавателей
            'niokr_total',              #  ОБЪЕМ НИОКР - активное участие в грантах, хоздоговорные работы
            'grants_per_100_npr',       #  ГРАНТЫ НА ПРЕПОДАВАТЕЛЯ - обучение подаче заявок, внутренние гранты
            'journals_published',       #  НАУЧНЫЕ ЖУРНАЛЫ - создание собственных журналов, индексируемых в базах
            'risc_publications'         #  ПУБЛИКАЦИИ РИНЦ - поддержка российских научных изданий
        ],
        "Финансовые показатели": [
            'total_income_per_student',  # ДОХОД НА СТУДЕНТА - платные услуги, эндаумент-фонды, коммерциализация разработок
            'self_income_per_npr',       #  ХОЗРАСЧЕТ НА ПРЕПОДАВАТЕЛЯ - коммерческие проекты, консалтинг
            'self_income_share',         #  ДОЛЯ ВНЕБЮДЖЕТНЫХ ДОХОДОВ - развитие платных образовательных услуг

        ],
        "Инфраструктура и кадры": [
            'npr_with_degree_percent',   #  ПРЕПОДАВАТЕЛИ С УЧЕНОЙ СТЕПЕНЬЮ - программы аспирантуры, поддержка защиты
            'young_npr_share',           #  МОЛОДЫЕ ПРЕПОДАВАТЕЛИ - программы привлечения молодых ученых
            'area_per_student',          #  ПЛОЩАДЬ НА СТУДЕНТА - строительство, реновация, эффективное использование
            'pc_per_student',            #  КОМПЬЮТЕРЫ НА СТУДЕНТА - обновление компьютерного парка
            'lib_books_per_student'      #  БИБЛИОТЕЧНЫЙ ФОНД - пополнение библиотек, электронные ресурсы
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
        user_df = user_df[feature_order]  # Убедимся в правильном порядке
        
        with st.spinner("Анализируем возможные улучшения..."):
            try:
                # Исправленный вызов метода suggest_improvement
                result = predictor.suggest_improvement(
                    user_df,
                    desired_top,
                    current_rank=current_rank,
                    allowed_features=allowed_features
                )
                
                # Распаковываем результат в зависимости от формата
                if len(result) == 2:
                    recommendations, improved_rank = result
                    percent_changes = []  # Создаем пустой список для процентных изменений
                else:
                    # Если метод возвращает три элемента
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
                        # Обрабатываем разные форматы рекомендаций
                        if len(recommendation) == 3:
                            feat, old, new = recommendation
                            # Вычисляем процентное изменение
                            if old > 0:
                                percent_change = ((new - old) / old * 100)
                            else:
                                percent_change = 100 if new > 0 else 0
                        elif len(recommendation) == 4:
                            feat, old, new, percent_change = recommendation
                        else:
                            continue  # Пропускаем некорректные рекомендации
                        
                        # Пропускаем незначимые изменения
                        if abs(percent_change) < 0.01 or abs(new - old) < 0.1:
                            continue
                        
                        meaningful_count += 1
                        col1, col2, col3 = st.columns([3, 2, 1])  # Исправлено: 3 колонки
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
        template_df.loc[0] = [0] * len(feature_order)  # Добавляем строку с нулями
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
        st.rerun()
    
    # Показать все необходимые признаки
    if st.checkbox("Показать все необходимые признаки"):
        st.write("**Всего признаков:**", len(feature_order))
        for i, feat in enumerate(feature_order, 1):
            st.write(f"{i}. {russian_name(feat)} ({feat})")