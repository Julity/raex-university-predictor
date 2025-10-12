#src\config.py
feature_order = [
    "egescore_avg", "egescore_contract", "egescore_min", "olympiad_winners", "olympiad_other",
    "target_admission_share", "competition", "magistracy_share", "aspirantura_share", "external_masters",
    "external_grad_share", "aspirants_per_100_students", "target_contract_in_tech",
    "foreign_students_share", "foreign_non_cis", "foreign_cis", "foreign_graduated",
    "mobility_outbound", "foreign_staff_share", "foreign_professors", "niokr_total",
    "niokr_share_total", "niokr_own_share", "niokr_per_npr", "npr_with_degree_percent",
    "npr_per_100_students", "ppc_salary_index", "avg_salary_grads", "scopus_publications",
    "risc_publications", "risc_citations", "foreign_niokr_income", "foreign_edu_income",
    "total_income_per_student", "self_income_per_npr", "self_income_share",
    "lib_books_per_student", "area_per_student", "pc_per_student", "young_npr_share",
    "journals_published", "grants_per_100_npr"
]

# config.py - ЗАМЕНИТЕ ВЕСА ПОЛНОСТЬЮ
feature_weights = {
    # ГРУППА 1: АКАДЕМИЧЕСКИЕ ПОКАЗАТЕЛИ (50%)
    "egescore_avg": 0.15,           # Самый важный академический показатель
    "egescore_contract": 0.08,
    "egescore_min": 0.05,
    "olympiad_winners": 0.12,       # Очень важно для топ-вузов
    "olympiad_other": 0.08,
    "competition": 0.10,
    
    # ГРУППА 2: НАУКА И ИССЛЕДОВАНИЯ (30%) - САМАЯ ВАЖНАЯ!
    "scopus_publications": 0.25,    # КРИТИЧЕСКИ ВАЖНО
    "niokr_total": 0.20,            # Очень важно
    "niokr_per_npr": 0.15,
    "risc_publications": 0.12,
    "risc_citations": 0.13,
    "niokr_share_total": 0.10,
    "niokr_own_share": 0.08,
    "grants_per_100_npr": 0.10,
    "journals_published": 0.08,
    "foreign_niokr_income": 0.07,
    
    # ГРУППА 3: МЕЖДУНАРОДНАЯ ДЕЯТЕЛЬНОСТЬ (10%)
    "foreign_students_share": 0.08,
    "foreign_non_cis": 0.06,
    "foreign_cis": 0.05,
    "mobility_outbound": 0.06,
    "foreign_staff_share": 0.05,
    "foreign_graduated": 0.04,
    "foreign_professors": 0.03,
    "foreign_edu_income": 0.05,
    
    # ГРУППА 4: ФИНАНСЫ И ИНФРАСТРУКТУРА (10%)
    "total_income_per_student": 0.07,
    "avg_salary_grads": 0.08,
    "self_income_per_npr": 0.05,
    "self_income_share": 0.04,
    "ppc_salary_index": 0.05,
    "npr_with_degree_percent": 0.06,
    "npr_per_100_students": 0.06,
    "area_per_student": 0.04,
    "lib_books_per_student": 0.03,
    "pc_per_student": 0.03,
    "young_npr_share": 0.04,
    
    # ГРУППА 5: МАГИСТРАТУРА И ЦЕЛЕВОЙ ПРИЕМ (менее важны)
    "target_admission_share": 0.04,
    "target_contract_in_tech": 0.03,
    "magistracy_share": 0.03,
    "aspirantura_share": 0.04,
    "external_masters": 0.02,
    "external_grad_share": 0.02,
    "aspirants_per_100_students": 0.04,
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

def russian_name(key: str) -> str:
    return russian_names.get(key, key)