# src/config.py
# config.py - УПОРЯДОЧЕННЫЙ СПИСОК ПРИЗНАКОВ

feature_order = [
    # Академические показатели (ядро)
    "egescore_avg", "egescore_contract", "egescore_min", 
    "olympiad_winners", "olympiad_other", "competition",
    
    # Целевой прием и магистратура
    "target_admission_share", "magistracy_share", "aspirantura_share",
    "external_masters", "external_grad_share", "aspirants_per_100_students",
    "target_contract_in_tech",
    
    # Международная деятельность
    "foreign_students_share", "foreign_non_cis", "foreign_cis",
    "foreign_graduated", "mobility_outbound", "foreign_staff_share", 
    "foreign_professors",
    
    # Научная деятельность
    "niokr_total", "niokr_share_total", "niokr_own_share", "niokr_per_npr",
    "scopus_publications", "risc_publications", "risc_citations",
    "foreign_niokr_income", "foreign_edu_income",
    
    # Финансовые показатели
    "total_income_per_student", "self_income_per_npr", "self_income_share",
    "ppc_salary_index", "avg_salary_grads",
    
    # Инфраструктура и кадры
    "npr_with_degree_percent", "npr_per_100_students", "young_npr_share",
    "lib_books_per_student", "area_per_student", "pc_per_student",
    
    # Дополнительные
    "journals_published", "grants_per_100_npr"
]

# Обновленные веса с нормализацией
# В config.py - РЕАЛИСТИЧНЫЕ ВЕСА RAEX
feature_weights = {
    # ЯДРО РЕЙТИНГА (70%)
    "egescore_avg": 0.15,           # Качество приема
    "avg_salary_grads": 0.15,       # Зарплаты выпускников  
    "scopus_publications": 0.12,    # Наука
    "niokr_total": 0.10,            # Исследования
    "olympiad_winners": 0.08,       # Таланты
    "foreign_students_share": 0.05, # Интернационализация
    "total_income_per_student": 0.05, # Ресурсы
    
    # ВТОРОСТЕПЕННЫЕ (25%)
    "competition": 0.04,
    "npr_with_degree_percent": 0.04, 
    "risc_citations": 0.03,
    "ppc_salary_index": 0.03,
    "foreign_edu_income": 0.03,
    "olympiad_other": 0.03,
    "niokr_per_npr": 0.02,
    "self_income_per_npr": 0.02,
    "grants_per_100_npr": 0.01,
    
    # МИНИМАЛЬНЫЕ (5%)
    "egescore_contract": 0.005,
    "egescore_min": 0.005,
    "area_per_student": 0.005,
    "npr_per_100_students": 0.005,
    "foreign_non_cis": 0.005,
    "mobility_outbound": 0.005,
    "pc_per_student": 0.005,
    "lib_books_per_student": 0.005
}
# Слабые признаки - минимальное влияние
weak_features = [
    'target_admission_share', 'magistracy_share', 'aspirantura_share',
    'external_masters', 'external_grad_share', 'aspirants_per_100_students',
    'target_contract_in_tech', 'foreign_cis', 'foreign_graduated', 
    'foreign_staff_share', 'foreign_professors', 'niokr_share_total',
    'niokr_own_share', 'self_income_share', 'lib_books_per_student',
    'young_npr_share', 'journals_published', 'foreign_niokr_income',
    'risc_publications'
]

# Слабым признакам нулевой вес
for feat in weak_features:
    if feat in feature_weights:
        feature_weights[feat] = 0.0


# Нормализация весов к 100%
total_weight = sum(feature_weights.values())
feature_weights = {k: v/total_weight for k, v in feature_weights.items()}

# РЕАЛЬНЫЕ ДИАПАЗОНЫ ИЗ СУЩЕСТВУЮЩИХ ДАННЫХ
realistic_ranges = {
    # Академические показатели (из данных: МГУ 90.0, слабые вузы ~60.0)
    'egescore_avg': (55.0, 95.0),
    'egescore_contract': (50.0, 85.0),
    'egescore_min': (45.0, 80.0),
    'olympiad_winners': (0, 300),      # МГУ: 220, слабые: 0
    'olympiad_other': (0, 300),       # МГУ: 230, слабые: 0
    'competition': (0.5, 20.0),       # МГУ: 7.7, слабые: ~0.1
    
    # Целевой прием и магистратура
    'target_admission_share': (0.5, 15.0),    # МГУ: 1.8, слабые: ~8-10
    'magistracy_share': (10.0, 35.0),         # МГУ: 28.0, слабые: ~5-15
    'aspirantura_share': (5.0, 40.0),         # МГУ: 35.0, слабые: ~2-10
    'external_masters': (30.0, 100.0),        # МГУ: 48.0
    'external_grad_share': (40.0, 70.0),      # МГУ: 64.0
    'aspirants_per_100_students': (0.5, 15.0), # МГУ: 11.0, слабые: ~0
    'target_contract_in_tech': (0.2, 5.0),    # МГУ: 1.2
    
    # Международная деятельность
    'foreign_students_share': (0.0, 25.0),    # МГУ: 15.8, РУДН: 15.5
    'foreign_non_cis': (0.0, 15.0),           # МГУ: 13.4
    'foreign_cis': (0.0, 10.0),               # МГУ: 1.5, РУДН: 3.5
    'foreign_graduated': (0.0, 25.0),         # МГУ: 19.5
    'mobility_outbound': (0.0, 2.0),          # МГУ: 0.5
    'foreign_staff_share': (0.0, 10.0),       # МГУ: 0.8
    'foreign_professors': (0, 100),           # МГУ: 75
    
    # Научная деятельность
    'niokr_total': (0, 12000000),             # МГУ: 9500000, слабые: ~0
    'niokr_share_total': (0.0, 30.0),         # МГУ: 23.0
    'niokr_own_share': (80.0, 100.0),         # МГУ: 99.0
    'niokr_per_npr': (0, 800),               # МГУ: 690.0
    'scopus_publications': (0, 3500),         # МГУ: 3200
    'risc_publications': (0, 200),           # МГУ: 150
    'risc_citations': (0, 3500),             # МГУ: 3100
    'foreign_niokr_income': (0, 300000),      # МГУ: 25000
    'foreign_edu_income': (0, 900000),        # МГУ: 800000
    
    # Финансовые показатели
    'total_income_per_student': (50000, 2500000), # МГУ: 1080.0 (тыс. руб)
    'self_income_per_npr': (0, 2500),         # МГУ: 2120.0
    'self_income_share': (10.0, 50.0),        # МГУ: 42.0
    'ppc_salary_index': (100.0, 200.0),       # МГУ: 185.0
    'avg_salary_grads': (30000, 120000),      # МГУ: 105000
    
    # Инфраструктура и кадры
    'npr_with_degree_percent': (50.0, 90.0),  # МГУ: 82.0
    'npr_per_100_students': (5.0, 25.0),      # МГУ: 17.0
    'young_npr_share': (5.0, 25.0),          # МГУ: 19.0
    'lib_books_per_student': (50, 300),       # МГУ: 250.0
    'area_per_student': (5.0, 30.0),         # МГУ: 22.0
    'pc_per_student': (0.3, 1.0),            # МГУ: 0.65
    
    # Дополнительные
    'journals_published': (0, 80),           # МГУ: 70
    'grants_per_100_npr': (0, 20.0)          # МГУ: 10.5
}

russian_names = {
    "egescore_avg": "Средний балл ЕГЭ",
    "egescore_contract": "Средний ЕГЭ на контракт",
    "egescore_min": "Минимальный ЕГЭ",
    "olympiad_winners": "Победители ВСОШ",
    "olympiad_other": "Прочие олимпиады",
    "target_admission_share": "Целевой приём, %",
    "competition": "Конкурс на 1 место",
    "magistracy_share": "Доля магистратуры, %",
    "aspirantura_share": "Доля аспирантуры, %",
    "external_masters": "Магистранты из других вузов, %",
    "external_grad_share": "Другие дипломы у магистров и аспирантов, %",
    "aspirants_per_100_students": "Аспирантов на 100 студентов",
    "target_contract_in_tech": "Целевики в тех. направлениях, %",
    "foreign_students_share": "Иностранные студенты, %",
    "foreign_non_cis": "Иностранцы вне СНГ, %",
    "foreign_cis": "Иностранцы из СНГ, %",
    "foreign_graduated": "Иностранцы, окончившие вуз, %",
    "mobility_outbound": "Стажировки за рубежом, %",
    "foreign_staff_share": "Иностранцы среди преподавателей, %",
    "foreign_professors": "Иностранные профессора",
    "niokr_total": "НИОКР всего, тыс. руб",
    "niokr_share_total": "НИОКР в доходе, %",
    "niokr_own_share": "Собственные НИОКР, %",
    "niokr_per_npr": "НИОКР на 1 НПР, тыс. руб",
    "npr_with_degree_percent": "Преподаватели с уч. степенью, %",
    "npr_per_100_students": "НПР на 100 студентов",
    "ppc_salary_index": "Зарплата ППС к средн. по региону, %",
    "avg_salary_grads": "Средняя зарплата выпускников, тыс. руб",
    "scopus_publications": "Публикации Scopus",
    "risc_publications": "Публикации в РИНЦ на 100 НПР",
    "risc_citations": "Цитирования в РИНЦ",
    "foreign_niokr_income": "Доходы от иностр. НИОКР, тыс. руб",
    "foreign_edu_income": "Доходы от иностр. студентов, тыс. руб",
    "total_income_per_student": "Доход на 1 студента, тыс. руб",
    "self_income_per_npr": "Хозрасчет на 1 НПР, тыс. руб",
    "self_income_share": "Хозрасчетные доходы, %",
    "lib_books_per_student": "Книг на 1 студента",
    "area_per_student": "Площадь на 1 студента, м²",
    "pc_per_student": "Компьютеров на 1 студента",
    "young_npr_share": "Молодые преподаватели, %",
    "journals_published": "Научные журналы",
    "grants_per_100_npr": "Грантов на 100 НПР"
}
# Добавить в config.py
weak_university_characteristics = {
    'egescore_avg': "ЕГЭ ниже 60 баллов",
    'scopus_publications': "Менее 100 публикаций Scopus", 
    'niokr_total': "НИОКР менее 500 тыс. руб.",
    'olympiad_winners': "Менее 20 победителей олимпиад",
    'foreign_students_share': "Менее 5% иностранных студентов",
    'avg_salary_grads': "Зарплата выпускников ниже 50 тыс. руб."
}

top_100_requirements = {
    'egescore_avg': "75+ баллов",
    'scopus_publications': "500+ публикаций",
    'niokr_total': "2+ млн руб.",
    'olympiad_winners': "50+ победителей", 
    'foreign_students_share': "10+% иностранцев",
    'avg_salary_grads': "70+ тыс. руб. зарплата"
}
def russian_name(key: str) -> str:
    return russian_names.get(key, key)

def validate_feature_range(feature: str, value: float) -> bool:
    """Проверяет, находится ли значение в реалистичном диапазоне"""
    if feature in realistic_ranges:
        min_val, max_val = realistic_ranges[feature]
        return min_val <= value <= max_val
    return True

def get_feature_statistics():
    """Возвращает статистику по признакам на основе реальных данных"""
    return {
        'egescore_avg': {'min': 55.0, 'max': 95.0, 'top_value': 90.0, 'weak_value': 60.0},
        'olympiad_winners': {'min': 0, 'max': 300, 'top_value': 220, 'weak_value': 0},
        'scopus_publications': {'min': 0, 'max': 3500, 'top_value': 3200, 'weak_value': 5},
        'foreign_students_share': {'min': 0.0, 'max': 25.0, 'top_value': 15.8, 'weak_value': 0.5},
        'niokr_total': {'min': 0, 'max': 12000000, 'top_value': 9500000, 'weak_value': 50000},
        'avg_salary_grads': {'min': 30000, 'max': 120000, 'top_value': 105000, 'weak_value': 35000}
    }
    # Метод преобразования для балльной системы
SCORING_METHOD = "raex_scores"  # Добавить эту строку

def transform_to_raex_scores(y):
    """Преобразует ранги в баллы по методике RAEX"""
    max_rank = 1000
    min_rank = 1
    # Линейное преобразование: ранг 1 -> 100 баллов, ранг 1000 -> 0 баллов
    scores = 100 * (max_rank - y) / (max_rank - min_rank)
    return scores

def scores_to_ranks(scores):
    """Преобразует баллы обратно в ранги"""
    max_rank = 1000
    min_rank = 1
    ranks = max_rank - (scores * (max_rank - min_rank) / 100)
    return ranks.round().astype(int)

# Реалистичные веса для балльной системы
feature_weights_scores = {
    # ЯДРО РЕЙТИНГА (70%)
    "egescore_avg": 0.15,           # Качество приема
    "avg_salary_grads": 0.15,       # Зарплаты выпускников  
    "scopus_publications": 0.12,    # Наука
    "niokr_total": 0.10,            # Исследования
    "olympiad_winners": 0.08,       # Таланты
    "foreign_students_share": 0.05, # Интернационализация
    "total_income_per_student": 0.05, # Ресурсы
    
    # ВТОРОСТЕПЕННЫЕ (25%)
    "competition": 0.04,
    "npr_with_degree_percent": 0.04, 
    "risc_citations": 0.03,
    "ppc_salary_index": 0.03,
    "foreign_edu_income": 0.03,
    "olympiad_other": 0.03,
    "niokr_per_npr": 0.02,
    "self_income_per_npr": 0.02,
    "grants_per_100_npr": 0.01,
    
    # МИНИМАЛЬНЫЕ (5%)
    "egescore_contract": 0.005,
    "egescore_min": 0.005,
    "area_per_student": 0.005,
    "npr_per_100_students": 0.005,
    "foreign_non_cis": 0.005,
    "mobility_outbound": 0.005,
    "pc_per_student": 0.005,
    "lib_books_per_student": 0.005
}

# Обновляем основные веса
feature_weights = feature_weights_scores

# Слабым признакам нулевой вес
for feat in weak_features:
    if feat in feature_weights:
        feature_weights[feat] = 0.0