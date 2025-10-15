# src/train_model.py
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBRegressor
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from config import feature_order, feature_weights, realistic_ranges, weak_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRankPredictor(nn.Module):
    def __init__(self, input_size):
        super(SimpleRankPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def add_synthetic_universities(df, n_top=200, n_mid=300, n_low=200):
    """Сбалансированные синтетические данные"""
    synthetic_data = []
    
    # ТОП-вузы (ранги 1-50) - ЛУЧШИЕ
    for i in range(n_top):
        synthetic_uni = {
            'egescore_avg': np.random.uniform(80, 95),
            'olympiad_winners': np.random.randint(50, 300),
            'olympiad_other': np.random.randint(100, 400),
            'competition': np.random.uniform(5.0, 15.0),
            'scopus_publications': np.random.randint(1000, 5000),
            'niokr_total': np.random.uniform(5000000, 15000000),
            'avg_salary_grads': np.random.uniform(80000, 150000),
            'total_income_per_student': np.random.uniform(800000, 2000000),
            'foreign_students_share': np.random.uniform(8.0, 25.0),
            'rank': np.random.randint(1, 51)  # ТОЛЬКО 1-50!
        }
        synthetic_uni = complete_synthetic_data(synthetic_uni, "top")
        synthetic_data.append(synthetic_uni)
    
    # Средние вузы (ранги 51-200)
    for i in range(n_mid):
        synthetic_uni = {
            'egescore_avg': np.random.uniform(65, 80),
            'olympiad_winners': np.random.randint(5, 50),
            'olympiad_other': np.random.randint(20, 100),
            'competition': np.random.uniform(2.0, 6.0),
            'scopus_publications': np.random.randint(100, 1000),
            'niokr_total': np.random.uniform(500000, 3000000),
            'avg_salary_grads': np.random.uniform(50000, 80000),
            'total_income_per_student': np.random.uniform(300000, 800000),
            'foreign_students_share': np.random.uniform(2.0, 10.0),
            'rank': np.random.randint(51, 201)  # ТОЛЬКО 51-200!
        }
        synthetic_uni = complete_synthetic_data(synthetic_uni, "mid")
        synthetic_data.append(synthetic_uni)
    
    # Слабые вузы (ранги 201-500) - ХУДШИЕ
    for i in range(n_low):
        synthetic_uni = {
            'egescore_avg': np.random.uniform(45, 60),
            'olympiad_winners': np.random.randint(0, 5),
            'olympiad_other': np.random.randint(0, 10),
            'competition': np.random.uniform(0.1, 2.0),
            'scopus_publications': np.random.randint(0, 50),
            'niokr_total': np.random.uniform(0, 200000),
            'avg_salary_grads': np.random.uniform(20000, 40000),
            'total_income_per_student': np.random.uniform(50000, 200000),
            'foreign_students_share': np.random.uniform(0.0, 2.0),
            'rank': np.random.randint(201, 501)  # ТОЛЬКО 201-500!
        }
        synthetic_uni = complete_synthetic_data(synthetic_uni, "low")
        synthetic_data.append(synthetic_uni)
    
    return pd.concat([df, pd.DataFrame(synthetic_data)], ignore_index=True)

def complete_synthetic_data(base_data, tier):
    """Заполняет остальные признаки в зависимости от tier"""
    defaults = {
        "top": {
            'egescore_contract': np.random.uniform(70, 85),
            'egescore_min': np.random.uniform(60, 75),
            'niokr_per_npr': np.random.uniform(500, 1500),
            'risc_citations': np.random.randint(2000, 5000),
            'risc_publications': np.random.randint(100, 300),
            'self_income_per_npr': np.random.uniform(1500, 3000),
            'ppc_salary_index': np.random.uniform(150, 250),
            'foreign_non_cis': np.random.uniform(5.0, 15.0),
            'foreign_edu_income': np.random.uniform(300000, 1000000),
            'npr_with_degree_percent': np.random.uniform(75, 90),
            'area_per_student': np.random.uniform(15, 30)
        },
        "mid": {
            'egescore_contract': np.random.uniform(60, 75),
            'egescore_min': np.random.uniform(50, 65),
            'niokr_per_npr': np.random.uniform(200, 600),
            'risc_citations': np.random.randint(500, 2000),
            'risc_publications': np.random.randint(50, 150),
            'self_income_per_npr': np.random.uniform(800, 1800),
            'ppc_salary_index': np.random.uniform(120, 180),
            'foreign_non_cis': np.random.uniform(1.0, 6.0),
            'foreign_edu_income': np.random.uniform(50000, 300000),
            'npr_with_degree_percent': np.random.uniform(65, 80),
            'area_per_student': np.random.uniform(10, 20)
        },
        "low": {
            'egescore_contract': np.random.uniform(50, 65),
            'egescore_min': np.random.uniform(40, 55),
            'niokr_per_npr': np.random.uniform(50, 250),
            'risc_citations': np.random.randint(0, 500),
            'risc_publications': np.random.randint(0, 50),
            'self_income_per_npr': np.random.uniform(200, 900),
            'ppc_salary_index': np.random.uniform(100, 140),
            'foreign_non_cis': np.random.uniform(0.0, 2.0),
            'foreign_edu_income': np.random.uniform(0, 50000),
            'npr_with_degree_percent': np.random.uniform(55, 70),
            'area_per_student': np.random.uniform(5, 15)
        }
    }
    
    for feat, value_range in defaults[tier].items():
        if feat not in base_data:
            if isinstance(value_range, tuple):
                base_data[feat] = np.random.uniform(value_range[0], value_range[1])
            else:
                base_data[feat] = value_range
    
    # Заполняем слабые признаки
    for feat in weak_features:
        if feat not in base_data:
            base_data[feat] = np.random.uniform(0, 10)  # Минимальные значения
    
    return base_data
def transform_target_variable(y):
    """Преобразует ранги в баллы RAEX (0-100)"""
    from config import transform_to_raex_scores
    return transform_to_raex_scores(y)

def inverse_transform_target(y_pred):
    """Преобразует баллы обратно в ранги (1-500)"""
    from config import scores_to_ranks
    return scores_to_ranks(y_pred)

def train_pytorch_model(X_train, y_train, X_val, y_val, epochs=500, patience=30):
    """Обучение нейросети PyTorch с ранней остановкой"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    input_size = X_train.shape[1]
    model = SimpleRankPredictor(input_size).to(device)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        if np.isnan(val_loss):
            continue
            
        scheduler.step(val_loss)
        
        if epoch % 50 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f'Ранняя остановка на эпохе {epoch}')
            break
    
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    return model

def load_and_preprocess_data(data_folder="data"):
    """Загрузка и предобработка данных"""
    df_list = []
    for file in glob.glob(f"{data_folder}/raex_*.csv"):
        df = pd.read_csv(file)
        if "year" not in df.columns:
            year = int(file.split("_")[-1].split(".")[0])
            df["year"] = year
        df_list.append(df)
    
    raw_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Загружено данных: {len(raw_df)} записей")
    logger.info(f"Годы в данных: {sorted(raw_df['year'].unique())}")
    
    # Добавляем синтетические данные
    logger.info("Добавление синтетических вузов...")
    raw_df = add_synthetic_universities(raw_df)
    logger.info(f"После добавления синтетических данных: {len(raw_df)} записей")
    
    return raw_df

def prepare_features(df):
    """Подготовка признаков БЕЗ применения весов"""
    X = df[feature_order].copy()
    
    # ТОЛЬКО обработка выбросов и заполнение NaN, БЕЗ весов!
    for feat in feature_order:
        upper_limit = X[feat].quantile(0.95)
        if upper_limit > 0:
            X[feat] = np.where(X[feat] > upper_limit, upper_limit, X[feat])
    
    X = X.fillna(X.median())
    
    return X

def transform_to_raex_scores(y):
    """Преобразует ранги в баллы по методике RAEX"""
    # RAEX: лучший вуз = ~100 баллов, худший = ~0 баллов
    max_rank = 500
    min_rank = 1
    
    # Линейное преобразование: ранг 1 -> 100 баллов, ранг 500 -> 0 баллов
    scores = 100 * (max_rank - y) / (max_rank - min_rank)
    return scores
def test_predictions(model, scaler):
    """Тестирование предсказаний"""
    logger.info("=== ТЕСТ ПРЕДСКАЗАНИЯ ===")
    
    test_cases = {
        'Топ-вуз (МГУ)': create_test_case("top"),
        'Средний вуз': create_test_case("mid"), 
        'Слабый вуз': create_test_case("low")
    }
    
    for name, test_data in test_cases.items():
        test_df = pd.DataFrame([test_data])
        test_df = test_df[feature_order]  # Правильный порядок
        test_scaled = scaler.transform(test_df)
        predicted_score = model.predict(test_scaled)[0]
        predicted_rank = inverse_transform_target(predicted_score)
        logger.info(f"{name}: предсказанный ранг {predicted_rank:.1f} (балл: {predicted_score:.1f})")

def test_real_universities(model, scaler):
    """Тест на реальных университетах"""
    logger.info("=== ТЕСТ РЕАЛЬНЫХ ВУЗОВ ===")
    
    # Данные МГТУ им. Баумана (2023)
    bmstu_data = {
        'egescore_avg': 80.83, 'egescore_contract': 71.98, 'egescore_min': 54.55,
        'olympiad_winners': 8, 'olympiad_other': 236, 'competition': 5.0,
        'target_admission_share': 13.59, 'target_contract_in_tech': 20.37,
        'magistracy_share': 10.30, 'aspirantura_share': 2.70,
        'external_masters': 98.72, 'external_grad_share': 47.70,
        'aspirants_per_100_students': 3.70,
        'foreign_students_share': 5.71, 'foreign_non_cis': 3.70, 'foreign_cis': 2.01,
        'foreign_graduated': 7.66, 'mobility_outbound': 0.07,
        'foreign_staff_share': 0.22, 'foreign_professors': 0,
        'niokr_total': 3982904.40, 'niokr_share_total': 22.40, 'niokr_own_share': 84.29,
        'niokr_per_npr': 1919.01, 'scopus_publications': 160.44, 'risc_publications': 160.44,
        'risc_citations': 409.68, 'foreign_niokr_income': 0.00, 'journals_published': 13,
        'grants_per_100_npr': 2.84,
        'foreign_edu_income': 31664.10, 'total_income_per_student': 827.28,
        'self_income_per_npr': 1939.98, 'self_income_share': 22.59,
        'ppc_salary_index': 200.57, 'avg_salary_grads': 100.0,
        'npr_with_degree_percent': 62.89, 'npr_per_100_students': 5.77,
        'young_npr_share': 13.63, 'lib_books_per_student': 106.41,
        'area_per_student': 10.36, 'pc_per_student': 0.36
    }
    
    test_df = pd.DataFrame([bmstu_data])
    test_df = test_df[feature_order]
    test_scaled = scaler.transform(test_df)
    predicted_score = model.predict(test_scaled)[0]
    predicted_rank = inverse_transform_target(predicted_score)
    
    logger.info(f"🎯 МГТУ им. Баумана: предсказанный ранг {predicted_rank:.1f} (балл: {predicted_score:.1f})")
    logger.info(f"📊 Реальный ранг Бауманки: 2 место")
def scores_to_ranks(scores):
    """Преобразует баллы обратно в ранги"""
    max_rank = 500
    min_rank = 1
    ranks = max_rank - (scores * (max_rank - min_rank) / 100)
    return ranks.round().astype(int)

def train_and_save_models(data_folder="data", model_folder="models"):
    """Обучение и сохранение моделей с балльной системой"""
    os.makedirs(model_folder, exist_ok=True)
    
    # Загрузка данных
    raw_df = load_and_preprocess_data(data_folder)
    
    # Подготовка признаков и целевой переменной
    X = prepare_features(raw_df)
    y = raw_df["rank"]
    
    # Преобразуем целевую переменную в БАЛЛЫ (0-100)
    y_transformed = transform_target_variable(y)
    
    # Детальная статистика
    logger.info("=== СТАТИСТИКА ДАННЫХ ===")
    logger.info(f"Записей: {len(raw_df)}")
    logger.info(f"Диапазон рангов: {y.min()} - {y.max()}")
    logger.info(f"Диапазон баллов: {y_transformed.min():.1f} - {y_transformed.max():.1f}")
    logger.info(f"Распределение рангов:")
    logger.info(f"Топ-50: {len(y[y <= 50])} вузов")
    logger.info(f"51-200: {len(y[(y > 50) & (y <= 200)])} вузов") 
    logger.info(f"201-500: {len(y[y > 200])} вузов")
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Масштабирование признаков
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("=== ОБУЧЕНИЕ XGBOOST (балльная система) ===")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=50
    )
    
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )
    
    # Предсказания и обратное преобразование в РАНГИ
    y_pred_train_scores = xgb_model.predict(X_train_scaled)
    y_pred_val_scores = xgb_model.predict(X_val_scaled)
    
    y_pred_train = inverse_transform_target(y_pred_train_scores)
    y_pred_val = inverse_transform_target(y_pred_val_scores)
    
    # Оригинальные ранги для метрик
    y_train_orig = inverse_transform_target(y_train)
    y_val_orig = inverse_transform_target(y_val)
    
    # Метрики
    train_mae = mean_absolute_error(y_train_orig, y_pred_train)
    train_r2 = r2_score(y_train_orig, y_pred_train)
    val_mae = mean_absolute_error(y_val_orig, y_pred_val)
    val_r2 = r2_score(y_val_orig, y_pred_val)
    
    logger.info(f"ОБУЧЕНИЕ - MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    logger.info(f"ВАЛИДАЦИЯ - MAE: {val_mae:.3f}, R²: {val_r2:.3f}")
    
    # Анализ ошибок по сегментам
    errors = y_pred_val - y_val_orig
    top_50_error = errors[y_val_orig <= 50].mean()
    mid_error = errors[(y_val_orig > 50) & (y_val_orig <= 200)].mean()
    low_error = errors[y_val_orig > 200].mean()
    
    logger.info(f"Ошибка по сегментам:")
    logger.info(f"Топ-50: {top_50_error:.3f}")
    logger.info(f"51-200: {mid_error:.3f}")
    logger.info(f"201-500: {low_error:.3f}")
    
    # Сохранение моделей
    dump(xgb_model, f"{model_folder}/xgb_model.pkl")
    dump(scaler, f"{model_folder}/scaler.pkl")
    
    # Анализ важности признаков
    feature_importance = pd.DataFrame({
        'feature': feature_order,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Топ-10 самых важных признаков:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # Сохранение информации с указанием метода
    model_info = {
        'feature_order': feature_order,
        'feature_weights': feature_weights,
        'training_years': sorted(raw_df['year'].unique()),
        'data_shape': X.shape,
        'performance': {
            'xgboost': {
                'train_mae': train_mae, 'train_r2': train_r2,
                'val_mae': val_mae, 'val_r2': val_r2
            }
        },
        'feature_importance': feature_importance.to_dict('records'),
        'target_transform': 'raex_scores',  # Указываем метод
        'target_stats': {
            'min_rank': y.min(),
            'max_rank': y.max(),
            'mean_rank': y.mean(),
            'min_score': y_transformed.min(),
            'max_score': y_transformed.max()
        },
        'model_type': 'xgboost'
    }
    
    dump(model_info, f"{model_folder}/model_info.pkl")
    dump('xgboost', f"{model_folder}/best_model_type.pkl")
    # После сохранения model_info добавь:
    logger.info("=== ПРОВЕРКА ВЕСОВ ===")
    logger.info(f"Используемые веса: {len(feature_weights)} признаков")
    top_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, weight in top_weights:
        logger.info(f"  {feat}: {weight:.4f}")
    logger.info("✅ Модель успешно обучена и сохранена! (балльная система)")
    
    # Тестирование на разных типах вузов
    test_predictions(xgb_model, scaler)
    
    # Дополнительный тест на реальных данных
    test_real_universities(xgb_model, scaler)
    # В конце train_and_save_models перед return
    import joblib
    saved_info = joblib.load(f"{model_folder}/model_info.pkl")
    print("=== ПРОВЕРКА ФАЙЛА model_info.pkl ===")
    print(f"Веса в файле: {len(saved_info['feature_weights'])} признаков")
    
def create_test_case(tier):
    """Создает тестовые данные для каждого tier со всеми признаками"""
    base_data = {
        'egescore_avg': 0.0, 'egescore_contract': 0.0, 'egescore_min': 0.0,
        'olympiad_winners': 0, 'olympiad_other': 0, 'competition': 0.0,
        'target_admission_share': 0.0, 'magistracy_share': 0.0, 'aspirantura_share': 0.0,
        'external_masters': 0.0, 'external_grad_share': 0.0, 'aspirants_per_100_students': 0.0,
        'target_contract_in_tech': 0.0, 'foreign_students_share': 0.0, 'foreign_non_cis': 0.0,
        'foreign_cis': 0.0, 'foreign_graduated': 0.0, 'mobility_outbound': 0.0,
        'foreign_staff_share': 0.0, 'foreign_professors': 0, 'niokr_total': 0.0,
        'niokr_share_total': 0.0, 'niokr_own_share': 0.0, 'niokr_per_npr': 0.0,
        'npr_with_degree_percent': 0.0, 'npr_per_100_students': 0.0, 'ppc_salary_index': 0.0,
        'avg_salary_grads': 0.0, 'scopus_publications': 0, 'risc_publications': 0.0,
        'risc_citations': 0.0, 'foreign_niokr_income': 0.0, 'foreign_edu_income': 0.0,
        'total_income_per_student': 0.0, 'self_income_per_npr': 0.0, 'self_income_share': 0.0,
        'lib_books_per_student': 0.0, 'area_per_student': 0.0, 'pc_per_student': 0.0,
        'young_npr_share': 0.0, 'journals_published': 0, 'grants_per_100_npr': 0.0
    }
    
    if tier == "top":
        # Топ-вуз (МГУ)
        base_data.update({
            'egescore_avg': 80.0, 'egescore_contract': 65.0, 'egescore_min': 60.0,
            'olympiad_winners': 220, 'olympiad_other': 230, 'competition': 7.7,
            'target_admission_share': 1.8, 'magistracy_share': 28.0, 'aspirantura_share': 35.0,
            'external_masters': 48.0, 'external_grad_share': 64.0, 'aspirants_per_100_students': 11.0,
            'target_contract_in_tech': 1.2, 'foreign_students_share': 15.8, 'foreign_non_cis': 13.4,
            'foreign_cis': 1.5, 'foreign_graduated': 19.5, 'mobility_outbound': 0.5,
            'foreign_staff_share': 0.8, 'foreign_professors': 75, 'niokr_total': 9500000,
            'niokr_share_total': 23.0, 'niokr_own_share': 99.0, 'niokr_per_npr': 690.0,
            'npr_with_degree_percent': 82.0, 'npr_per_100_students': 17.0, 'ppc_salary_index': 185.0,
            'avg_salary_grads': 105000, 'scopus_publications': 3200, 'risc_publications': 150,
            'risc_citations': 3100, 'foreign_niokr_income': 25000, 'foreign_edu_income': 800000,
            'total_income_per_student': 1080000, 'self_income_per_npr': 2120.0, 'self_income_share': 42.0,
            'lib_books_per_student': 250, 'area_per_student': 22.0, 'pc_per_student': 0.65,
            'young_npr_share': 19.0, 'journals_published': 70, 'grants_per_100_npr': 10.5
        })
    elif tier == "mid":
        # Средний вуз
        base_data.update({
            'egescore_avg': 70.0, 'egescore_contract': 60.0, 'egescore_min': 50.0,
            'olympiad_winners': 20, 'olympiad_other': 50, 'competition': 3.0,
            'target_admission_share': 5.0, 'magistracy_share': 20.0, 'aspirantura_share': 15.0,
            'external_masters': 35.0, 'external_grad_share': 45.0, 'aspirants_per_100_students': 5.0,
            'target_contract_in_tech': 1.0, 'foreign_students_share': 5.0, 'foreign_non_cis': 3.0,
            'foreign_cis': 2.0, 'foreign_graduated': 8.0, 'mobility_outbound': 0.2,
            'foreign_staff_share': 1.0, 'foreign_professors': 10, 'niokr_total': 1000000,
            'niokr_share_total': 15.0, 'niokr_own_share': 90.0, 'niokr_per_npr': 300.0,
            'npr_with_degree_percent': 70.0, 'npr_per_100_students': 10.0, 'ppc_salary_index': 130.0,
            'avg_salary_grads': 60000, 'scopus_publications': 300, 'risc_publications': 80,
            'risc_citations': 800, 'foreign_niokr_income': 5000, 'foreign_edu_income': 100000,
            'total_income_per_student': 500000, 'self_income_per_npr': 1000.0, 'self_income_share': 30.0,
            'lib_books_per_student': 150, 'area_per_student': 15.0, 'pc_per_student': 0.5,
            'young_npr_share': 15.0, 'journals_published': 10, 'grants_per_100_npr': 5.0
        })
    else:  # low
        # Слабый вуз
        base_data.update({
            'egescore_avg': 48.0, 'egescore_contract': 45.0, 'egescore_min': 40.0,
            'olympiad_winners': 0, 'olympiad_other': 1, 'competition': 0.3,
            'target_admission_share': 10.0, 'magistracy_share': 10.0, 'aspirantura_share': 5.0,
            'external_masters': 20.0, 'external_grad_share': 30.0, 'aspirants_per_100_students': 1.0,
            'target_contract_in_tech': 0.5, 'foreign_students_share': 1.0, 'foreign_non_cis': 0.5,
            'foreign_cis': 0.5, 'foreign_graduated': 2.0, 'mobility_outbound': 0.05,
            'foreign_staff_share': 0.2, 'foreign_professors': 2, 'niokr_total': 50000,
            'niokr_share_total': 5.0, 'niokr_own_share': 80.0, 'niokr_per_npr': 100.0,
            'npr_with_degree_percent': 60.0, 'npr_per_100_students': 5.0, 'ppc_salary_index': 100.0,
            'avg_salary_grads': 35000, 'scopus_publications': 10, 'risc_publications': 20,
            'risc_citations': 100, 'foreign_niokr_income': 0, 'foreign_edu_income': 10000,
            'total_income_per_student': 150000, 'self_income_per_npr': 300.0, 'self_income_share': 15.0,
            'lib_books_per_student': 80, 'area_per_student': 8.0, 'pc_per_student': 0.3,
            'young_npr_share': 10.0, 'journals_published': 1, 'grants_per_100_npr': 1.0
        })
    
    return base_data

    
if __name__ == "__main__":
    train_and_save_models()