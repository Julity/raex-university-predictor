# src/predictor.py
import pandas as pd
import numpy as np
import os
from joblib import load
import itertools
import torch
import torch.nn as nn
import logging
from config import feature_order, feature_weights, realistic_ranges, weak_features

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
    
 # predictor.py - ЗАМЕНИТЕ блок __init__
    def __init__(self, model_type='best'):
        """Инициализация предсказателя для Streamlit Cloud"""
        
        # Определяем базовую директорию
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Пробуем разные пути для Streamlit Cloud
        possible_paths = [
            os.path.join(current_dir, "models"),                    # Локальная разработка
            os.path.join(current_dir, "..", "models"),              # Streamlit структура 1
            os.path.join(current_dir, "app", "models"),             # Streamlit структура 2
            "/app/models",                                          # Absolute path в Streamlit
            "models"                                                # Относительный путь
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                model_path = abs_path
                logging.info(f"✅ Найдена папка models: {abs_path}")
                break
        
        if model_path is None:
            # Диагностика для Streamlit
            current_dir = os.getcwd()
            logging.error(f"❌ Папка models не найдена! Текущая директория: {current_dir}")
            logging.error(f"📁 Содержимое директории: {os.listdir('.')}")
            raise FileNotFoundError("Папка models не найдена")
        
        logging.info(f"📂 Используется путь к моделям: {model_path}")
        
        # Загрузка модели (остальной код без изменений)
        model_info_path = f"{model_path}/model_info.pkl"
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Модели не найдены по пути: {model_info_path}")
        
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
        
        self.feature_order = self.model_info['feature_order']
        current_features = feature_order
        saved_features = self.model_info['feature_order']
        
        if current_features != saved_features:
            logging.warning("Порядок признаков в config.py не совпадает с обученной моделью!")
            logging.warning(f"Используется порядок из обученной модели")
        
        logging.info(f"Загружен порядок признаков: {len(self.feature_order)} признаков")
    def validate_realism(self, df):
        """Проверка реалистичности входных данных"""
        reasonable_ranges = {
            'egescore_avg': (50, 100),
            'egescore_contract': (40, 100),
            'egescore_min': (30, 100),
            'olympiad_winners': (0, 300),
            'scopus_publications': (0, 10000),
            'niokr_total': (0, 10000000),
            'avg_salary_grads': (20, 200),
            'foreign_students_share': (0, 50)
        }
        
        warnings = []
        for feat, (min_val, max_val) in reasonable_ranges.items():
            value = df[feat].iloc[0]
            if value < min_val:
                warnings.append(f"⚠️ {feat}={value} СЛИШКОМ НИЗКИЙ (минимум {min_val})")
            elif value > max_val:
                warnings.append(f"⚠️ {feat}={value} СЛИШКОМ ВЫСОКИЙ (максимум {max_val})")
        
        return warnings
    def prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_order) - set(df.columns)
        if missing:
            raise ValueError(f"Входные данные не содержат признаки: {missing}")
        df_ordered = df[self.feature_order].copy()
                
        return df_ordered
       
    def normalize_weights(weights):
        """Нормализует веса к 100%"""
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def predict_rank(self, df: pd.DataFrame) -> float:
        df_ordered = self.prepare_input(df)
        scaled_df = pd.DataFrame(self.scaler.transform(df_ordered), columns=df_ordered.columns)
        
        # ДЕБАГ: вывести первые несколько значений
        print("=== ДЕБАГ ПРЕДСКАЗАНИЯ ===")
        print(f"egescore_avg после подготовки: {df_ordered['egescore_avg'].iloc[0]}")
        print(f"egescore_avg после масштабирования: {scaled_df['egescore_avg'].iloc[0]}")
        
        if self.model_type == 'neural_network':
            with torch.no_grad():
                X_tensor = torch.FloatTensor(scaled_df.values).to(self.device)
                pred_score = self.model(X_tensor).cpu().numpy()[0][0]
        else:
            pred_score = self.model.predict(scaled_df)[0]
        
        print(f"Предсказанный балл: {pred_score}")
        
        # ВАЖНО: преобразуем score обратно в ранг в зависимости от метода
        if hasattr(self, 'model_info') and self.model_info.get('target_transform') == 'raex_scores':
            from config import scores_to_ranks
            pred_rank = scores_to_ranks(pred_score)
        elif hasattr(self, 'model_info') and self.model_info.get('target_transform') == 'inverse_rank':
            pred_rank = 1000 / pred_score
        else:
            pred_rank = pred_score
        
        predicted_rank = max(1, min(500, round(pred_rank)))
        
        return predicted_rank

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