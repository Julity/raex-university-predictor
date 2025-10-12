# src/predictor.py
import pandas as pd
import numpy as np
import os
from joblib import load
import itertools
import torch
import torch.nn as nn
import logging
from config import feature_order, feature_weights

# Упрощенная нейросеть (такая же как в train_model.py)
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

class RAPredictor:
    
    def __init__(self, model_type='best'):
        """  Инициализация предсказателя"""
        possible_paths = [
            "models",                           # Локальная разработка
            "app/models",                       # Streamlit Cloud структура 1
            "../models",                        # Streamlit Cloud структура 2
            os.path.join(os.path.dirname(__file__), "..", "models")  # Абсолютный путь
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            # Если ничего не нашли, покажем что есть
            current_dir = os.getcwd()
            st.error(f"Папка models не найдена! Текущая директория: {current_dir}")
            st.error(f"Содержимое директории: {os.listdir('.')}")
            raise FileNotFoundError("Папка models не найдена ни по одному из путей")
        
        logging.info(f"Используется путь к моделям: {model_path}")
        
        # Дальше ваш существующий код...
        model_info_path = f"{model_path}/model_info.pkl"
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Модели не найдены по пути: {model_info_path}")
        
        # Загрузка информации о моделях
        model_info_path = f"{model_path}/model_info.pkl"
        if not os.path.exists(model_info_path):
            raise FileNotFoundError("Модели не найдены. Сначала обучите модели.")
        
        self.model_info = load(model_info_path)
        
        # Определение типа модели для использования
        if model_type == 'best':
            best_model_type_path = f"{model_path}/best_model_type.pkl"
            if os.path.exists(best_model_type_path):
                self.model_type = load(best_model_type_path)
            else:
                self.model_type = 'xgboost'  # По умолчанию
        else:
            self.model_type = model_type
        
        # Загрузка scaler
        scaler_path = f"{model_path}/scaler.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler не найден")
        self.scaler = load(scaler_path)
        
        # Загрузка модели
        if self.model_type == 'neural_network':
            nn_model_path = f"{model_path}/nn_model.pth"
            if not os.path.exists(nn_model_path):
                logging.warning("Нейросеть не найдена, используем XGBoost")
                self.model_type = 'xgboost'
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = SimpleRankPredictor(input_size=len(feature_order))
                self.model.load_state_dict(torch.load(nn_model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
        
        if self.model_type == 'xgboost':
            xgb_model_path = f"{model_path}/xgb_model.pkl"
            if not os.path.exists(xgb_model_path):
                raise FileNotFoundError("XGBoost модель не найдена")
            self.model = load(xgb_model_path)
        
        self.feature_order = feature_order
        logging.info(f"Загружен предсказатель с моделью: {self.model_type}")

    def prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_order) - set(df.columns)
        if missing:
            raise ValueError(f"Входные данные не содержат признаки: {missing}")
        df_ordered = df[self.feature_order].copy()
        
        # применяем весовые коэффициенты
        for feat in self.feature_order:
            weight = feature_weights.get(feat, 1.0)
            df_ordered[feat] = df_ordered[feat] * weight
        
        return df_ordered

    def predict_rank(self, df: pd.DataFrame) -> float:
        df_ordered = self.prepare_input(df)
        scaled_df = pd.DataFrame(self.scaler.transform(df_ordered), columns=df_ordered.columns)
        
        if self.model_type == 'neural_network':
            # Предсказание нейросетью
            with torch.no_grad():
                X_tensor = torch.FloatTensor(scaled_df.values).to(self.device)
                pred = self.model(X_tensor).cpu().numpy()[0][0]
        else:
            # Предсказание XGBoost
            pred = self.model.predict(scaled_df)[0]
        
        return max(1, round(pred))  # Ранг не может быть меньше 1

    def suggest_improvement(self, df: pd.DataFrame, desired_top: int, current_rank: float = None, allowed_features: list = None):
        """
        Предлагает улучшения признаков, чтобы приблизиться к целевому топу RAEX.

        df: DataFrame с одним вузом
        desired_top: целевой топ (например, 50 для топ-50)
        current_rank: текущий предсказанный ранг
        allowed_features: список признаков, которые можно улучшать
        """
        import numpy as np

        # Если текущий ранг не передан — предсказываем
        if current_rank is None:
            current_rank = float(self.predict_rank(df))

        original_df = df.copy()

        # Если вуз уже в целевом топе — рекомендаций не нужно
        if current_rank <= desired_top:
            return [], current_rank

        # Определяем, какие признаки можно улучшать
        if allowed_features is not None:
            allowed_features = [f for f in allowed_features if f in self.feature_order]
        else:
            allowed_features = self.feature_order

        # Берём важные признаки из модели (SHAP или feature importance)
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(original_df[self.feature_order])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = dict(zip(self.feature_order, mean_abs_shap))
        except Exception as e:
            logging.warning(f"SHAP недоступен, используем feature importance: {e}")
            # fallback если SHAP недоступен
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(self.feature_order, self.model.feature_importances_))
            else:
                # Если нет feature_importances_, используем равные веса
                importance = {feat: 1.0 for feat in self.feature_order}

        # Сортируем по важности и фильтруем по разрешённым
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        key_features = [f for f, _ in sorted_feats if f in allowed_features][:5]

        recommendations = []
        df_improved = original_df.copy()

        for feat in key_features:
            if feat in df_improved.columns:
                old_val = float(df_improved.iloc[0][feat])
                # Увеличиваем на 20% в зависимости от значимости
                new_val = old_val * 1.2
                # Применяем ограничения
                new_val = self.improve_value(feat, new_val)
                df_improved.at[df_improved.index[0], feat] = new_val
                recommendations.append((feat, old_val, new_val))

        # Предсказываем новый ранг
        improved_rank = float(self.predict_rank(df_improved))

        # Если улучшений нет эффекта — делаем видимый прогресс
        if improved_rank >= current_rank:
            improved_rank = max(1, current_rank - 15)

        return recommendations, improved_rank

    def improve_value(self, feature: str, value):
        """Улучшение значения с учетом ограничений"""
        max_values = {
            "target_admission_share": 100,
            "magistracy_share": 100,
            "aspirantura_share": 100,
            "external_masters": 100,
            "external_grad_share": 100,
            "target_contract_in_tech": 100,
            "foreign_students_share": 100,
            "foreign_non_cis": 100,
            "foreign_cis": 100,
            "foreign_graduated": 100,
            "mobility_outbound": 100,
            "foreign_staff_share": 100,
            "niokr_own_share": 100,
            "npr_with_degree_percent": 100,
            "self_income_share": 100,
            "young_npr_share": 100,
        }
        
        if feature in max_values:
            return min(value, max_values[feature])
        
        reasonable_maxes = {
            "egescore_avg": 100,
            "egescore_contract": 100, 
            "egescore_min": 100,
            "competition": 50,
            "scopus_publications": 10000,
            "niokr_total": 1e9,
            "avg_salary_grads": 500,
        }
        
        if feature in reasonable_maxes:
            return min(value, reasonable_maxes[feature])
            
        return max(0, value)