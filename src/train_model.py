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

from config import feature_order, feature_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Упрощенная нейросеть
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

def train_pytorch_model(X_train, y_train, X_val, y_val, epochs=500, patience=30):
    """Обучение нейросети PyTorch с ранней остановкой"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    input_size = X_train.shape[1]
    model = SimpleRankPredictor(input_size).to(device)
    
    # Инициализация весов
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    criterion = nn.HuberLoss(delta=1.0)  # Более устойчивая функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Преобразование данных в тензоры
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
        # Обучение
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Проверка на NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss на эпохе {epoch}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Проверка на NaN
        if np.isnan(val_loss):
            logger.warning(f"NaN val loss на эпохе {epoch}")
            continue
            
        scheduler.step(val_loss)
        
        if epoch % 50 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        # Ранняя остановка
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
        logger.warning("Не удалось обучить нейросеть. Возвращаем случайную модель.")
        best_model_state = model.state_dict().copy()
    
    # Загрузка лучшей модели
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
    
    # Проверка на NaN и бесконечные значения
    logger.info(f"NaN значений в признаках: {raw_df[feature_order].isna().sum().sum()}")
    logger.info(f"Бесконечных значений: {np.isinf(raw_df[feature_order].values).sum()}")
    
    return raw_df

def prepare_features(df):
    """Подготовка признаков с весами и обработкой выбросов"""
    X = df[feature_order].copy()
    
    # Обработка выбросов - ограничение 99 перцентилем
    for feat in feature_order:
        upper_limit = X[feat].quantile(0.99)
        if upper_limit > 0:
            X[feat] = np.where(X[feat] > upper_limit, upper_limit, X[feat])
    
    # Применяем весовые коэффициенты
    for feat in feature_order:
        weight = feature_weights.get(feat, 1.0)
        X[feat] = X[feat] * weight
    
    # Заполнение NaN (если есть)
    X = X.fillna(X.median())
    
    return X

def train_and_save_models(data_folder="data", model_folder="models"):
    """Обучение и сохранение моделей с улучшенными параметрами"""
    os.makedirs(model_folder, exist_ok=True)
    
    # Загрузка данных
    raw_df = load_and_preprocess_data(data_folder)
    
    # Подготовка признаков и целевой переменной
    X = prepare_features(raw_df)
    y = raw_df["rank"]
    
    # Детальная статистика
    logger.info("=== СТАТИСТИКА ДАННЫХ ===")
    logger.info(f"Записей: {len(raw_df)}")
    logger.info(f"Диапазон рангов: {y.min()} - {y.max()}")
    logger.info(f"Средний ранг: {y.mean():.2f}")
    logger.info(f"Медианный ранг: {y.median():.2f}")
    
    # Проверим распределение рангов
    rank_counts = y.value_counts().sort_index()
    logger.info("Распределение рангов:")
    for rank in sorted(rank_counts.index[:10]):  # Топ-10
        logger.info(f"Ранг {rank}: {rank_counts[rank]} вузов")
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Масштабирование признаков
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("=== ОБУЧЕНИЕ XGBOOST ===")
    # Улучшенные параметры XGBoost
    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        early_stopping_rounds=100
    )
    
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )
    
    # Предсказания
    y_pred_train = xgb_model.predict(X_train_scaled)
    y_pred_val = xgb_model.predict(X_val_scaled)
    
    # Метрики
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    logger.info(f"ОБУЧЕНИЕ - MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    logger.info(f"ВАЛИДАЦИЯ - MAE: {val_mae:.3f}, R²: {val_r2:.3f}")
    
    # Анализ ошибок
    errors = y_pred_val - y_val
    logger.info(f"Средняя ошибка: {errors.mean():.3f}")
    logger.info(f"Std ошибок: {errors.std():.3f}")
    
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
    
    # Сохранение информации
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
        'feature_importance': feature_importance.to_dict('records')
    }
    
    dump(model_info, f"{model_folder}/model_info.pkl")
    dump('xgboost', f"{model_folder}/best_model_type.pkl")
    
    logger.info("✅ Модель успешно обучена и сохранена!")
    
    # Проверим предсказание для типичного топ-вуза
    logger.info("=== ТЕСТ ПРЕДСКАЗАНИЯ ===")
    top_university_data = X.mean().to_frame().T  # Средние значения
    top_university_scaled = scaler.transform(top_university_data)
    predicted_rank = xgb_model.predict(top_university_scaled)[0]
    logger.info(f"Предсказанный ранг для среднего вуза: {predicted_rank:.1f}")

if __name__ == "__main__":
    train_and_save_models()