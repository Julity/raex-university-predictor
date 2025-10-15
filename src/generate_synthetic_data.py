# src/generate_synthetic_data.py
import pandas as pd
import numpy as np
import os

def generate_comprehensive_synthetic_data():
    """Генерирует полный набор синтетических данных для обучения"""
    
    synthetic_data = []
    
    # 1. Топ-вузы (ранги 1-50)
    for i in range(50):
        uni = {
            'university': f'Топ_Вуз_{i+1}',
            'year': 2024,
            'rank': i + 1,
            'egescore_avg': np.random.uniform(80, 95),
            'egescore_contract': np.random.uniform(70, 85),
            'egescore_min': np.random.uniform(60, 80),
            'olympiad_winners': np.random.randint(50, 300),
            'olympiad_other': np.random.randint(100, 500),
            'competition': np.random.uniform(5, 20),
            'target_admission_share': np.random.uniform(1, 10),
            'magistracy_share': np.random.uniform(20, 50),
            'aspirantura_share': np.random.uniform(10, 35),
            'external_masters': np.random.uniform(30, 80),
            'external_grad_share': np.random.uniform(40, 70),
            'aspirants_per_100_students': np.random.uniform(5, 15),
            'target_contract_in_tech': np.random.uniform(0.5, 5),
            'foreign_students_share': np.random.uniform(5, 25),
            'foreign_non_cis': np.random.uniform(3, 15),
            'foreign_cis': np.random.uniform(2, 10),
            'foreign_graduated': np.random.uniform(10, 30),
            'mobility_outbound': np.random.uniform(0.5, 5),
            'foreign_staff_share': np.random.uniform(1, 10),
            'foreign_professors': np.random.randint(10, 100),
            'niokr_total': np.random.uniform(1e6, 1e7),
            'niokr_share_total': np.random.uniform(15, 35),
            'niokr_own_share': np.random.uniform(80, 99),
            'niokr_per_npr': np.random.uniform(500, 2000),
            'npr_with_degree_percent': np.random.uniform(70, 95),
            'npr_per_100_students': np.random.uniform(8, 20),
            'ppc_salary_index': np.random.uniform(150, 250),
            'avg_salary_grads': np.random.uniform(80000, 150000),
            'scopus_publications': np.random.randint(500, 5000),
            'risc_publications': np.random.uniform(100, 500),
            'risc_citations': np.random.uniform(1000, 10000),
            'foreign_niokr_income': np.random.uniform(50000, 500000),
            'foreign_edu_income': np.random.uniform(100000, 1000000),
            'total_income_per_student': np.random.uniform(500000, 2000000),
            'self_income_per_npr': np.random.uniform(1000, 5000),
            'self_income_share': np.random.uniform(30, 60),
            'lib_books_per_student': np.random.uniform(50, 200),
            'area_per_student': np.random.uniform(15, 30),
            'pc_per_student': np.random.uniform(0.5, 1.5),
            'young_npr_share': np.random.uniform(15, 35),
            'journals_published': np.random.randint(5, 50),
            'grants_per_100_npr': np.random.uniform(10, 50),
        }
        synthetic_data.append(uni)
    
    # 2. Средние вузы (ранги 51-200) - аналогично, с меньшими значениями
    # 3. Слабые вузы (ранги 201-500) - с очень низкими значениями
    
    return pd.DataFrame(synthetic_data)

if __name__ == "__main__":
    df = generate_comprehensive_synthetic_data()
    df.to_csv("data/synthetic_universities.csv", index=False)
    print(f"Сгенерировано {len(df)} синтетических записей")